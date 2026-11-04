# GPT-OSS-120B (uncensored) on Runpod Serverless — scale-to-zero, OpenAI-compatible

Self-hosted, OpenAI-SDK-compatible inference for
[`HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive)
(a derestricted fine-tune of OpenAI's open-weight `gpt-oss-120b`, 117B total /
~5.1B active params per token, MoE, Apache-2.0, distributed as a 61GB
GGUF/MXFP4 file), running on Runpod Serverless with GPU workers that scale
from **0 → 1 → 0** automatically.

## 1. Architecture

```
Client (OpenAI SDK)
   │  POST https://<your-proxy>/v1/chat/completions
   ▼
proxy/ (FastAPI gateway)          <- holds RUNPOD_API_KEY, never exposed to client
   │  POST https://api.runpod.io/v2/<endpoint>/run|runsync + poll /stream|/status
   ▼
Runpod Serverless Endpoint         <- Flex workers, workersMin=0 (Active Workers = 0)
   │  idle -> 0 GPU workers, no billing
   │  request arrives -> Runpod starts 1 worker automatically
   ▼
Worker container (Dockerfile, src/handler.py)
   │  on cold start: download GGUF to the network volume if missing,
   │  then start llama-server once and keep it warm
   ▼
llama-server (llama.cpp, CUDA)  --  OpenAI-compatible /v1/chat/completions,
   serving the GGUF loaded from  /runpod-volume/models/...  (persistent
   network volume — never re-downloaded on later cold starts)
```

**Why llama.cpp instead of vLLM:** this specific model is published only as
a GGUF/MXFP4 file. vLLM's GGUF loader does not reliably support MoE models
in MXFP4 (gpt-oss architecture) — it's explicitly experimental for that
combination. `llama-server` has first-class, actively-maintained support for
exactly this format and already exposes an OpenAI-compatible API with
streaming, so it was chosen as the serving runtime per the project's
"if vLLM can't serve it, pick the most suitable runtime" rule.

**Why a network volume instead of baking the model into the image:** the
model is 61GB. Baking it into the Docker image would make every `docker
pull` on every new worker 61GB, and any image rebuild (e.g. a handler.py bug
fix) would force a full model re-upload/re-download. A persistent network
volume mounted at `/runpod-volume` is downloaded into once and reused by
every worker that starts afterwards, cold or warm — only the ~2-3GB
container image needs to be pulled per worker.

## 2. GPU choice

| | |
|---|---|
| Model weights (GGUF/MXFP4) | 61 GB on disk |
| MoE active params | ~5.1B per token (fast inference despite 117B total) |
| KV cache + runtime overhead | several more GB, scales with context length and concurrency |
| **Chosen GPU** | **NVIDIA A100 80GB / H100 80GB** (PCIe or SXM, whichever Runpod has available) |

61GB of weights plus KV cache does not fit on anything smaller than an
80GB-class GPU without offloading MoE experts to CPU (`N_CPU_MOE`, supported
by `llama-server` and exposed as an env var here if you need to downgrade to
a 48GB GPU at the cost of throughput). 80GB GPUs were chosen over
multi-GPU tensor-parallel setups for simplicity (1 `gpuCount`) and because
Runpod's A100/H100-80GB availability is generally good.

Indicative Runpod on-demand serverless pricing (verify current rates in the
Runpod console/pricing page before relying on these — they change):

| GPU | ~$/hr (flex, active) | 5 min | 10 min | 30 min | 60 min |
|---|---|---|---|---|---|
| A100 80GB | ~$1.80–2.50/hr | ~$0.15–0.21 | ~$0.30–0.42 | ~$0.90–1.25 | ~$1.80–2.50 |
| H100 80GB | ~$3.00–4.50/hr | ~$0.25–0.38 | ~$0.50–0.75 | ~$1.50–2.25 | ~$3.00–4.50 |

At `workersMin=0`, you only pay for the seconds a worker is actually
running (cold start + generation + the `idleTimeout` tail), never for idle
GPU time with no worker. The network volume itself has its own small,
continuous storage cost (see §8) regardless of whether a worker is running.

## 3. Repository layout

```
.
├── Dockerfile                  # worker image: llama.cpp (CUDA) + handler.py
├── requirements.txt            # worker Python deps
├── src/handler.py              # Runpod handler: downloads model once, runs
│                                #   llama-server, forwards/streams requests
├── proxy/                      # OpenAI-compatible gateway (deploy anywhere)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── deploy/
│   └── create_infra.py         # creates Runpod network volume + template + endpoint via API
├── scripts/
│   ├── download_model.sh       # optional pre-warm of the network volume
│   ├── test_endpoint.sh        # scale-to-zero end-to-end test
│   └── benchmark.py            # TTFT / tokens-per-sec / cold vs warm, via OpenAI SDK
├── .github/workflows/docker-build.yml   # builds+pushes the worker image (CI has the disk a laptop may not)
├── .env.example
└── .gitignore
```

## 4. Prerequisites you need to provide

- A **Runpod account + API key** (Settings → API Keys in the Runpod console).
- A **Docker Hub account** (or another registry — adjust the workflow/image
  references) with a repo to push `gptoss-120b-worker` to, and an access
  token for `docker login`.
- Nothing is required from Hugging Face: the model is public, Apache-2.0,
  no gating. `HF_TOKEN` in `.env.example` is only there for the future, if
  you ever swap in a gated model.

None of these are stored anywhere in this repository — put them in a local
`.env` (gitignored) or your CI/host's secret store.

## 5. Build and push the worker image

The CUDA build stage is multi-gigabyte; building on a disk-constrained
machine is not recommended. The included GitHub Actions workflow does it on
GitHub's runners instead:

1. Push this repo to GitHub.
2. In the repo's Settings → Secrets and variables → Actions, add:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN` (a Docker Hub access token, not your password)
3. Push to `main` (or run the workflow manually from the Actions tab). It
   builds `Dockerfile` and pushes:
   ```
   docker.io/<DOCKERHUB_USERNAME>/gptoss-120b-worker:latest
   docker.io/<DOCKERHUB_USERNAME>/gptoss-120b-worker:<git-sha>
   ```

To build locally instead (needs real disk — don't do this on an
8-10GB-free machine):
```bash
docker build -t docker.io/<user>/gptoss-120b-worker:latest .
docker push docker.io/<user>/gptoss-120b-worker:latest
```

## 6. Create the Runpod endpoint

```bash
pip install httpx
export RUNPOD_API_KEY=your-runpod-api-key
python3 deploy/create_infra.py \
  --image docker.io/<user>/gptoss-120b-worker:latest \
  --name gptoss-120b \
  --data-center US-KS-2 \
  --volume-size-gb 100 \
  --gpu-type-ids "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB PCIe"
```

This creates, in order: a Network Volume (persistent model storage), a
Serverless Template (your image + env vars), and a Serverless Endpoint with
`workersMin=0` (Active Workers = 0), `workersMax=1`, Flex workers,
`idleTimeout=60`, `flashboot=true`, attached to the volume. Re-running the
script reuses whatever it already created (tracked in
`deploy/infra_state.json` — delete that file to force fresh resources).

Copy the printed `RUNPOD_ENDPOINT_ID` into your proxy's environment.

If you'd rather click through the UI: Runpod console → Serverless → New
Endpoint → pick your template/image → GPU = A100/H100 80GB → **Active
Workers = 0**, **Max Workers = 1**, enable **Flex Workers**, **Idle
Timeout ≈ 60s**, attach the network volume. The API script above does
exactly this.

## 7. Pre-warm the model (recommended, optional)

`src/handler.py` downloads the 61GB GGUF to the network volume automatically
on its very first cold start ever — so technically you can skip this step
and just send a request. But that first real user request would then wait
through the full download *and* model load, which can be tens of minutes.
To avoid that:

```bash
# Spin up a cheap CPU/GPU Runpod Pod with the SAME network volume attached
# at /workspace (or any path), then inside that pod:
MODEL_DIR=/workspace/models/gpt-oss-120b-uncensored ./scripts/download_model.sh
# Terminate the pod once it finishes — you only pay for pod time, not storage you already had.
```

Make sure the path your Pod wrote to matches `MODEL_DIR` as seen by the
*serverless* worker (`/runpod-volume/models/gpt-oss-120b-uncensored` — the
mount point is `/runpod-volume` for serverless workers specifically, which
may differ from the mount path you chose for the Pod).

## 8. Deploy the proxy

The proxy is a small stateless FastAPI app — deploy it wherever you like
(your own VM, Fly.io, Railway, a container platform, etc.). It needs:

```bash
cd proxy
cp ../.env.example .env   # fill in RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, PROXY_API_KEY
docker build -t gptoss-proxy .
docker run --env-file .env -p 8000:8000 gptoss-proxy
```

`PROXY_API_KEY` is the key *your* clients will use — it is unrelated to
`RUNPOD_API_KEY`, which never leaves this service. Put a real domain + TLS
(e.g. behind nginx/Caddy or your platform's built-in HTTPS) in front of it
before exposing it publicly, and keep `ALLOWED_ORIGINS` empty or scoped to
your own frontend's origin (never `*`) if browsers will call it directly.

## 9. Test

**curl:**
```bash
curl https://your-proxy.example.com/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b-uncensored","messages":[{"role":"user","content":"Hello"}]}'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(api_key="<PROXY_API_KEY>", base_url="https://your-proxy.example.com/v1")

response = client.chat.completions.create(
    model="gpt-oss-120b-uncensored",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)

with client.chat.completions.stream(
    model="gpt-oss-120b-uncensored",
    messages=[{"role": "user", "content": "Напиши короткое приветствие"}],
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
```

**Scale-to-zero cycle** (the thing this whole project exists to prove):
```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
./scripts/test_endpoint.sh
```
It checks worker count = 0, sends a request, times the cold start, checks
worker count ≥ 1, sends a warm request (much faster), waits past
`idleTimeout`, checks worker count is back to 0, then sends one more request
and confirms the worker auto-starts again.

**Latency/throughput benchmark:**
```bash
pip install openai
PROXY_BASE_URL=https://your-proxy.example.com/v1 PROXY_API_KEY=... \
  python3 scripts/benchmark.py
```
Reports cold vs warm total latency, time-to-first-token, and tokens/sec.

## 10. Logs

- Worker logs (model download, llama-server startup, per-request
  cold/warm, TTFT, generation timing) — Runpod console → your endpoint →
  Workers → select a worker → Logs, or `runpod` CLI / API logs endpoint.
- Proxy logs — stdout of wherever you deployed it (`docker logs`, your
  platform's log viewer). Prompts are **not** logged by default
  (`LOG_PROMPTS=false`); flip it to `true` only if you need to debug content
  and understand the privacy implications of doing so.

## 11. Verify scale-to-zero yourself

Runpod console → Serverless → your endpoint → the **Workers** tab shows the
live worker count. Leave it idle past `idleTimeout` and confirm it drops to
0; send a request and confirm it goes back to 1. `scripts/test_endpoint.sh`
automates this exact check.

## 12. Updating the model

- Same file, new weights: just delete the file at
  `/runpod-volume/models/gpt-oss-120b-uncensored/...` (e.g. via a short-lived
  Pod with the volume attached) and the next cold start re-downloads it.
- Different model/repo entirely: update `MODEL_REPO` / `MODEL_FILE` /
  `SERVED_MODEL_NAME` in the template's env (re-run
  `deploy/create_infra.py` after deleting the relevant keys from
  `deploy/infra_state.json`, or edit the template via the Runpod console/API),
  and make sure the new file is also GGUF-compatible with llama-server (or
  switch back to the vLLM path if the new model has solid vLLM support).

## 13. Tearing it down

```bash
curl -X DELETE https://rest.runpod.io/v1/endpoints/<endpoint_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/templates/<template_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/network-volumes/<volume_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
```
IDs are in `deploy/infra_state.json`. Deleting the network volume is what
stops the ongoing storage charge — the endpoint and template alone don't
cost anything when idle.

## 14. Cost summary

| Resource | Billed when idle? |
|---|---|
| GPU worker (`workersMin=0`) | **No** — $0 when no job is running, scale-to-zero confirmed by `scripts/test_endpoint.sh` |
| Network volume (model storage) | **Yes** — flat per-GB-month storage fee regardless of traffic (check current rate in the Runpod console for your chosen size/tier) |
| Docker Hub / GitHub Actions | Free tier is enough for one private-ish image at this scale |

## 15. Known limitations / what to double-check before production

- `llama-server` CLI flags (`--n-cpu-moe`, `--flash-attn`, etc.) can change
  between llama.cpp releases — the Dockerfile builds `master` by default so
  the first build succeeds; pin `LLAMA_CPP_REF` to a tested release tag
  afterwards and re-verify flags with `llama-server --help` in that build.
  See the comment above `ARG LLAMA_CPP_REF` in `Dockerfile`.
  [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
- The exact Runpod GPU type strings (`gpuTypeIds`) and data center codes can
  change; `deploy/create_infra.py` will surface a clear 4xx error from the
  API if a value is stale — check the current list in the Runpod console.
- This was built and logic-tested against a local mock of the Runpod job
  API (proxy translation, SSE streaming, auth, non-stream path all verified
  working end-to-end). It has **not** been run against the real Runpod
  endpoint or the real 61GB model yet — that requires the Runpod API key
  and Docker Hub credentials only you can provide (see §16 / final report).
