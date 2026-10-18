# protocol.py patch: tolerate `tool_choice` with no tools

## Why

vLLM's `ChatCompletionRequest` rejects any request that carries `tool_choice` without `tools`:

```
When using `tool_choice`, `tools` must be set. (parameter=tool_choice)
```

That is right for a named tool or `"required"` — both are contradictions when nothing can be called.
It is wrong for `"auto"`, whose meaning is *"use a tool if one is available"*. With no tools supplied
there is nothing to choose from, so the field is a no-op and the request is really just a plain
completion.

Codex's remote-compaction request is exactly that shape: a summarisation with no tools. The 400 kills
the compaction, Codex reports `remote compact task` failure, and Multica ends the whole delegated task
as `agent_error.unknown`. Observed as 97 failures out of ~5,400 requests on this deployment, in the
95k-252k prompt range where compaction actually fires.

## What it changes

Inserted after the empty-array check and before the `"auto"` defaulting:

```python
if ("tools" not in data or data["tools"] is None) and data.get(
    "tool_choice"
) in ("auto", "none"):
    data.pop("tool_choice", None)
```

- `"auto"` / `"none"` with no tools -> field dropped, request proceeds
- named tool or `"required"` with no tools -> **still raises**, unchanged
- `tools: []` -> **still raises** the empty-array error, unchanged
- any request that does carry tools -> untouched

## Verifying

```bash
# compaction shape: must return text, not 400
curl -s localhost:8899/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-Flash-Local","max_tokens":40,"tool_choice":"auto",
  "messages":[{"role":"user","content":"Summarize: we set KV to 24 GiB."}]}'

# contradiction: must still 400
curl -s localhost:8899/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-Flash-Local","max_tokens":10,"tool_choice":"required",
  "messages":[{"role":"user","content":"hi"}]}'
```

## Regenerating after an image update

```bash
docker run --rm --entrypoint sh -v "$PWD:/out" eugr/spark-vllm-b12x:latest -c \
  'cp /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py /out/protocol.orig.py'
# then re-apply the insert and write protocol.py
```

`serve-fp4-tp2.sh` mounts `protocol.py` only when the file exists, so deleting it reverts to stock
behaviour without touching the image.
