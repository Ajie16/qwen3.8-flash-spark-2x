#!/usr/bin/env python3
"""Functional verification for the FP4 + B12X service.

Walks the checks in order and exits non-zero on the first hard failure, so it works as a deploy gate
as well as a debugging aid:

  1. /v1/models lists the expected ids
  2. basic generation returns non-empty text
  3. thinking can be turned off and on per request
  4. streaming produces incremental chunks
  5. /metrics exposes the spec-decode counters
  6. MTP acceptance, per content type
  7. an image is accepted (skipped unless --vision)

Every check prints a pass/fail line. The acceptance report is informational: a low number is a
property of the content, not a failure, so it never affects the exit code.

    python3 verify.py                          # against http://10.0.0.1:8899
    python3 verify.py --api http://host:8899 --model Qwen3.8-Flash-Local
    python3 verify.py --vision                 # also send a 1x1 PNG
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request

DEFAULT_API = "http://10.0.0.1:8899"
DEFAULT_MODEL = "Qwen3.8-Flash-Local"

# 1x1 transparent PNG. Enough to prove the multimodal path accepts image content at all.
PNG_1X1 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

ACCEPT_CASES = {
    "count (most predictable)": "Count from 1 to 200. Output only the numbers.",
    "json": "Output a JSON invoice with seller, buyer, ten line items, taxes and totals.",
    "code": "Write a Python implementation of a thread-safe LRU cache with get/put, tests included.",
    "prose-en": "Explain how a hash map works, covering collisions, resizing and complexity.",
    "technical-zh": "详细解释一致性哈希的原理、虚拟节点、扩容时的数据迁移过程。",
    "prose-zh": "写一篇关于城市夜间经济的说明文，包括消费结构、就业带动和治理难点。",
}

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print("  [%s] %-46s %s" % (mark, name, detail))
    if not ok:
        _failures.append(name)
    return ok


def post(api: str, body: dict, timeout: float = 600):
    req = urllib.request.Request(api + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def generate(api: str, model: str, prompt, max_tokens: int = 64,
             thinking: bool | None = None, stream: bool = False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    if stream:
        body["stream"] = True
        chunks = 0
        with post(api, body) as r:
            for raw in r:
                if raw.decode("utf-8", "replace").strip().startswith("data:"):
                    chunks += 1
        return chunks
    with post(api, body) as r:
        return json.load(r)


def spec_counters(api: str) -> tuple[float, float]:
    try:
        with urllib.request.urlopen(api + "/metrics", timeout=30) as r:
            text = r.read().decode()
    except Exception:
        return 0.0, 0.0

    def grab(field: str) -> float:
        m = re.search(r"vllm:spec_decode_%s_total\{[^}]*\}\s+([0-9.e+]+)" % field, text)
        return float(m.group(1)) if m else 0.0

    return grab("num_draft_tokens"), grab("num_accepted_tokens")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--vision", action="store_true", help="also exercise the image path")
    ap.add_argument("--accept-tokens", type=int, default=500,
                    help="max tokens per acceptance case; longer gives tighter rates")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="acceptance sampling temperature; the model default is 1.0")
    args = ap.parse_args()

    print("target %s  model %s\n" % (args.api, args.model))

    # ── 1. reachable and serving the expected id ──────────────────────────────
    try:
        with urllib.request.urlopen(args.api + "/v1/models", timeout=20) as r:
            served = [m["id"] for m in json.load(r).get("data", [])]
        check("/v1/models", True, ", ".join(served))
        check("model id present", args.model in served,
              args.model if args.model in served else "got %s" % served)
    except Exception as exc:
        check("/v1/models", False, "%s: %s" % (type(exc).__name__, exc))
        print("\nservice unreachable -- is the container up?")
        return 1

    # ── 2. basic generation ──────────────────────────────────────────────────
    try:
        d = generate(args.api, args.model, "Reply with the single word: ready", 16, thinking=False)
        text = (d["choices"][0]["message"].get("content") or "").strip()
        check("generation returns text", bool(text), repr(text[:40]))
    except Exception as exc:
        check("generation returns text", False, "%s: %s" % (type(exc).__name__, exc))

    # ── 3. thinking toggles per request ──────────────────────────────────────
    try:
        off = generate(args.api, args.model, "What is 17*23? One number.", 64, thinking=False)
        on = generate(args.api, args.model, "What is 17*23? One number.", 256, thinking=True)
        off_reason = (off["choices"][0]["message"].get("reasoning")
                      or off["choices"][0]["message"].get("reasoning_content") or "")
        on_reason = (on["choices"][0]["message"].get("reasoning")
                     or on["choices"][0]["message"].get("reasoning_content") or "")
        check("thinking off yields no reasoning", not off_reason, "%d chars" % len(off_reason))
        check("thinking on yields reasoning", bool(on_reason), "%d chars" % len(on_reason))
    except Exception as exc:
        check("thinking toggle", False, "%s: %s" % (type(exc).__name__, exc))

    # ── 4. streaming ─────────────────────────────────────────────────────────
    try:
        n = generate(args.api, args.model, "Count to 20.", 128, thinking=False, stream=True)
        check("streaming yields chunks", n > 1, "%d data frames" % n)
    except Exception as exc:
        check("streaming yields chunks", False, "%s: %s" % (type(exc).__name__, exc))

    # ── 5. metrics ───────────────────────────────────────────────────────────
    draft, accept = spec_counters(args.api)
    check("/metrics exposes spec-decode counters", draft > 0,
          "draft_since_start=%.0f accepted=%.0f" % (draft, accept))

    # ── 6. vision (opt-in) ───────────────────────────────────────────────────
    if args.vision:
        try:
            content = [{"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + PNG_1X1}},
                       {"type": "text", "text": "Describe this image in one word."}]
            d = generate(args.api, args.model, content, 32, thinking=False)
            text = (d["choices"][0]["message"].get("content") or "").strip()
            check("image accepted", bool(text), repr(text[:40]))
        except Exception as exc:
            check("image accepted", False, "%s: %s" % (type(exc).__name__, exc))

    # ── 7. acceptance, informational ─────────────────────────────────────────
    print("\n  MTP acceptance (temperature %.1f, %d tokens per case)"
          % (args.temperature, args.accept_tokens))
    print("  %-26s %8s %9s %8s" % ("case", "drafts", "accepted", "rate"))
    total_d = total_a = 0
    for name, prompt in ACCEPT_CASES.items():
        d0, a0 = spec_counters(args.api)
        try:
            generate(args.api, args.model, prompt, args.accept_tokens, thinking=False)
        except Exception as exc:
            print("  %-26s  FAILED %s" % (name, str(exc)[:40]))
            continue
        d1, a1 = spec_counters(args.api)
        dd, da = d1 - d0, a1 - a0
        total_d += dd
        total_a += da
        rate = 100.0 * da / dd if dd else 0.0
        print("  %-26s %8d %9d %7.1f%%" % (name, int(dd), int(da), rate))
    if total_d:
        print("  %-26s %8d %9d %7.1f%%" % ("overall", int(total_d), int(total_a),
                                           100.0 * total_a / total_d))
    print("\n  Low Chinese numbers are expected: the MTP head was trained mostly on English.")
    print("  No configuration knob moves this -- see docs/fp4-b12x-deployment.md.")

    print()
    if _failures:
        print("FAILED %d check(s): %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
