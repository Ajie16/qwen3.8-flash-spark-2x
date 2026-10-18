# Qwen3.8-Flash-Next on two DGX Sparks — FP4 + vLLM B12X

Serving recipe for `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local` (and the official
`nvidia/Qwen3.8-Flash-Next-FP4-Local` it derives from) on **two NVIDIA DGX Spark nodes** with
**vLLM + the B12X image**, tensor-parallel across the CX-7 RoCE link.

Measured on this hardware: **105 tok/s single stream, 426 at 8 concurrent**, prefill **2,300–2,700
tok/s** flat from 5k to 250k tokens.

Sibling repo for the single-node routes: [`qwen3.8-flash-spark-1x`](https://github.com/Ajie16/qwen3.8-flash-spark-1x)
covers EXL3 + TabbyAPI and TensorFold + MLX, plus the full deployment write-up this one shares.

## Why this checkpoint needs this specific image

`Qwen3.8-Flash-Next-FP4-Local` carries **147,456 `nvfp4_csf_*` scale tensors** — a compressed scale
format ("CSF") that upstream vLLM does not decode. Only `eugr/spark-vllm-b12x` has it:

| image | CSF-aware files | can load it |
|---|---|---|
| **`eugr/spark-vllm-b12x`** | **15** | ✅ |
| `vllm/vllm-openai:qwen38-flash-next` | 0 | ❌ |
| `ghcr.io/anemll/dspark-vllm-gx10` | 0 | ❌ (older build) |

The older image is correct for the `RadixArk/…-NVFP4` checkpoint, which is a *different format*, not
an older one.

## ★ Three settings that decide whether it works

**① `--device /dev/infiniband:/dev/infiniband` (plus `--cap-add SYS_NICE`)**

Without it the container cannot see `uverbs`/`rdma_cm`, **NCCL silently falls back to TCP**, and
throughput drops from 426 to 281 tok/s. The tell is `roceP2p1s0f0`'s `port_xmit_data` counter staying
flat — `ss` is misleading, because the out-of-band channel is TCP even when RDMA works.

**② `VLLM_ENABLE_ROCE_ALLREDUCE=1` + `VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`**

Generic NCCL collectives cost ~40 µs each with ~122 per token, which held TP=2 to **+23%** over one
node. vLLM's own RoCE path takes it to **+75%**. Pair it with `draft_tensor_parallel_size: 2` so the
MTP drafter also spans both ranks.

**③ `--mamba-ssm-cache-dtype float32`**

The GDN kernel in B12X asserts `state_dtype must be torch.float32`. Setting bfloat16 (which works on
stock vLLM) aborts the workers. Pinned deliberately.

## Files

| file | purpose |
|---|---|
| `serve-fp4-tp2.sh` | launcher. `0` = head (serves the API), `1` = worker (headless). Start the worker first. |
| `tf-restart.sh` | one command: sync → stop → drop caches → clear `/dev/shm` → start → wait → report. Use this. |
| **`verify.py`** | **deploy gate.** Checks reachability, model id, generation, thinking on/off, streaming, metrics, vision, and reports acceptance. Exits non-zero on failure. |
| `bench.py` | `decode` / `prefill` / `accept` benchmark modes. |
| `serve-fp4-tp1.sh`, `serve-fp4-local.sh` | single-node variants. |
| `pull_via_proxy.py`, `fetch_hf_repo.py` | image and checkpoint fetch helpers. |
| `qwen3.8-flash-next-nvfp4-cluster.yaml` | the upstream reference recipe this is based on. |
| `apply_tuning_patch.py`, `b12x_startup.py.patch` | **archived dead end** — see below. |

## Configuration

Every site-specific value is overridable, with a generic default:

```bash
HEAD_IP=10.0.0.1                # rank 0 on the RoCE link
REMOTE_HOST=node0               # ssh alias for rank 0
MODEL_DIR=$HOME/models/Qwen3.8-Flash-Next-FP4-Local
SUDO_PASS_FILE=               # optional; unset = root steps skipped
```

`tf-restart.sh` drops the page cache and clears `/dev/shm` because **vLLM compares
`--gpu-memory-utilization` against `MemFree`, not `MemAvailable`**, so leftovers from a previous run
read as unavailable memory and abort the start with
`Free memory on device (80.46/121.69 GiB) on startup is less than desired`. Those two leaks had
together cost up to 55 GiB.

Note `--ipc host` makes `--shm-size` a no-op — the container uses the host's `/dev/shm` tmpfs.

## Measured

Decode, clock locked at 2000 MHz, `nbt=8192`, thinking off, warm-up discarded:

| concurrency | aggregate tok/s | per stream |
|---|---|---|
| 1 | 105.0 | 105.0 |
| 4 | 281.8 | 70.4–71.9 |
| 8 | 426.1 | 53.3–54.8 |

Prefill 2,359–2,687 tok/s from 31k to 251k, flat.

MTP acceptance varies widely by content: **98% counting, ~70% code and JSON, ~52% English prose,
~27% Chinese prose** — the last is a property of the MTP head's training data and no configuration
knob moves it (LM-head precision and probabilistic drafting both tested, both no-ops).

## Bottleneck

Both ends are hard, and neither is tuning.

**Single stream (105 tok/s, 44 ms per engine step)** is bound by reading the ~590 MB of expert weights
that step activates, amortised over only 4.8 tokens. Clock elasticity is 0.20 — cutting the SM clock
9.5% costs 1.9%, which is what "memory-bound" looks like.

**High concurrency** saturates memory bandwidth: at C8 the read is ~248 GB/s against a measured peak
of 246 GB/s. An 8× batch costs only 2.1× per step, which is exactly why throughput scales 3.9×.

## Archived: the nbt "boundary"

This repo previously carried a patch for
`RuntimeError: preparation ranks reached incompatible tuning boundaries`, on the belief that
`--max-num-batched-tokens` could not exceed 4096. **That was wrong.** 8192 and 16384 both start
cleanly with no patch and zero mismatch warnings; the failures were a rank-to-rank race and retrying
the same value can pass.

`apply_tuning_patch.py` is kept only as a record of what was tried. **Do not apply it.** 4096 remains
a defensible choice for memory reasons (each doubling costs 6–7 GiB per node for 3–7% prefill), but
8192 is what this recipe ships.

## License

**The code in this repository is MIT** -- see [LICENSE](LICENSE). That covers the launch scripts, the
restart wrapper, the benchmark and verification tools, and the documentation.

**That is not the whole story, and it matters.** The MIT grant applies only to what is in this repo.
It does not extend to anything you point it at:

- **The checkpoint.** `nvidia/Qwen3.8-Flash-Next-FP4-Local` is distributed under the **NVIDIA Open
  Model Agreement**, and the base model under the **Qwen Community License**. Both carry their own
  terms for use and redistribution.
- **The abliterated variant** additionally derives from
  `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`, which was itself produced from
  `Qwen/Qwen3.8-Flash-Next` through the Unsloth GGUF pipeline, plus a delta from
  `unsloth/Qwen3.8-Flash-Next-GGUF`. Read all of those before deploying.
- **The image.** `eugr/spark-vllm-b12x` bundles vLLM (Apache-2.0) and NVIDIA's B12X components, which
  have their own licenses.

No model weights are redistributed here. The upstream reference recipe
(`qwen3.8-flash-next-nvfp4-cluster.yaml`) is copied verbatim from `eugr/spark-vllm-docker` for
reference and remains under that project's terms.
