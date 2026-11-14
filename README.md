# GPT-OSS-120B (uncensored) — scale-to-zero, OpenAI-compatible serverless

Self-hosted, OpenAI-SDK-compatible inference for
[`HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive)
(a derestricted fine-tune of OpenAI's open-weight `gpt-oss-120b`, 117B total /
~5.1B active params per token, MoE, Apache-2.0, distributed as a 61GB
GGUF/MXFP4 file), on GPU workers that scale from **0 → 1 → 0** automatically.

**Active deployment path: [Modal](https://modal.com).** The project
originally targeted Runpod Serverless (see
[Appendix A](#appendix-a-runpod-serverless-alternativelegacy-path)); the
worker image, build pipeline and runtime logic were fully built and verified
there, but Runpod requires a **prepaid account balance (≥ $5)** to create a
network volume, which the user could not add. Modal has the same
scale-to-zero / pay-per-second / custom-Docker-image model with **no prepaid
minimum**, so the deployment moved there. The worker image, model, and
llama.cpp runtime are unchanged — only the orchestration layer differs.

## 1. Architecture

```
Client (OpenAI SDK)
   │  POST https://<workspace>--gptoss-120b-worker-web.modal.run/v1/chat/completions
   ▼
Modal Function (ASGI gateway, src/modal_gateway.py)
   │  checks PROXY_API_KEY, rate-limits, restricts CORS
   │  (same container, loopback — no network hop, unlike the Runpod proxy)
   ▼
llama-server (llama.cpp, CUDA)  --  OpenAI-compatible /v1/chat/completions,
   serving the GGUF loaded from a persistent Modal Volume mounted at /cache
   (never re-downloaded on later cold starts)
   ▲
Modal autoscaler: min_containers=0 (Active Workers = 0, scale-to-zero)
   <-> 1 GPU container spun up automatically when a request arrives
```

One container does both jobs — gateway and inference — because they're on
the same machine anyway; this is simpler than Runpod's design (separate
proxy service translating into a job-queue API) and was possible specifically
*because* Modal's `@modal.asgi_app()` can sit directly in front of a
sidecar process on localhost.

**Why llama.cpp instead of vLLM:** this specific model is published only as
a GGUF/MXFP4 file. vLLM's GGUF loader does not reliably support MoE models
in MXFP4 (gpt-oss architecture) — it's explicitly experimental for that
combination. `llama-server` has first-class, actively-maintained support for
exactly this format and already exposes an OpenAI-compatible API with
streaming, so it was chosen as the serving runtime per the project's
"if vLLM can't serve it, pick the most suitable runtime" rule.

**Why a persistent volume instead of baking the model into the image:** the
model is 61GB. Baking it into the Docker image would make every cold start
pull 61GB, and any image rebuild (e.g. a handler bug fix) would force a full
model re-upload/re-download. A Modal Volume mounted at `/cache` is written to
once (first cold start ever) and reused by every container that starts
afterwards — only the ~2-3GB container image needs to be pulled per
container.

## 2. GPU choice

| | |
|---|---|
| Model weights (GGUF/MXFP4) | 61 GB on disk |
| MoE active params | ~5.1B per token (fast inference despite 117B total) |
| KV cache + runtime overhead | several more GB, scales with context length and concurrency |
| **Chosen GPU** | **A100-80GB / H100** (`MODAL_GPU_TYPES`, priority order) |

61GB of weights plus KV cache does not fit on anything smaller than an
80GB-class GPU without offloading MoE experts to CPU (`N_CPU_MOE`, supported
by `llama-server` and exposed as an env var here if you need to downgrade to
a 48GB GPU at the cost of throughput). 80GB GPUs were chosen over
multi-GPU tensor-parallel setups for simplicity (1 GPU per container).

Indicative Modal per-second pricing (verify current rates at
[modal.com/pricing](https://modal.com/pricing) before relying on these —
they change):

| GPU | $/sec | 5 min | 10 min | 30 min | 60 min |
|---|---|---|---|---|---|
| A100 80GB | $0.000694 | $0.21 | $0.42 | $1.25 | $2.50 |
| H100 SXM5 | $0.001097 | $0.33 | $0.66 | $1.98 | $3.95 |

At `min_containers=0`, you only pay for the seconds a container is actually
running (cold start + generation + the `scaledown_window` idle tail), never
for idle GPU time with no container. The Modal Volume has no separate
storage line item in Modal's pricing (unlike Runpod's network volume).

## 3. Repository layout

```
.
├── Dockerfile                  # worker image: llama.cpp (CUDA) + src/*.py (shared by both paths)
├── requirements.txt            # worker Python deps (Runpod path; Modal installs its own at deploy time)
├── modal_app.py                 # ACTIVE: Modal App definition (GPU, volume, secrets, autoscaling)
├── src/
│   ├── llama_runtime.py        # shared: download model once, launch+health-check llama-server
│   ├── modal_gateway.py        # ACTIVE: FastAPI gateway used inside the Modal container (auth/rate-limit/CORS)
│   └── handler.py              # Runpod path: generator handler, forwards/streams requests
├── deploy/
│   ├── modal_deploy.sh         # ACTIVE: modal token/secret setup + `modal deploy`
│   └── create_infra.py         # Runpod path: creates network volume + template + endpoint via API
├── proxy/                      # Runpod path: standalone OpenAI-compatible gateway (deploy anywhere)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── download_model.sh       # Runpod path: optional pre-warm of the network volume
│   ├── test_endpoint.sh        # Runpod path: scale-to-zero end-to-end test
│   └── benchmark.py            # TTFT / tokens-per-sec / cold vs warm, via OpenAI SDK (works against either path's base_url)
├── .github/workflows/docker-build.yml   # builds+pushes the worker image (CI has the disk a laptop may not)
├── .env.example
└── .gitignore
```

## 4. Prerequisites you need to provide

- A **Modal account** ([modal.com](https://modal.com) — sign up yourself;
  Claude cannot create accounts or enter payment details on your behalf) and
  an **API token**: Modal dashboard → Settings → API Tokens → "New Token",
  giving you a token ID + token secret.
- A **Docker Hub account** with a repo to push `gptoss-120b-worker` to (this
  part is already done — see §5), and an access token for the private pull
  (Modal needs a `REGISTRY_USERNAME`/`REGISTRY_PASSWORD` secret to pull a
  private image; a **Read-only** Docker Hub token is enough for this, unlike
  the Read & Write one needed to *push*).
- Nothing is required from Hugging Face: the model is public, Apache-2.0,
  no gating. `HF_TOKEN` is only there for the future, if you ever swap in a
  gated model.

None of these are stored anywhere in this repository — they live in a local
`.env` (gitignored) and, at deploy time, in Modal's own encrypted Secret
store (never baked into the image or committed to git).

## 5. Build and push the worker image

Already done for this deployment — see commit history and the
[`docker-build.yml`](.github/workflows/docker-build.yml) CI run. The worker
image at `docker.io/<DOCKERHUB_USERNAME>/gptoss-120b-worker:latest` is used
unchanged by both the Modal and Runpod paths. To rebuild after a code change,
push to `main` (or run the workflow manually from the Actions tab); it
builds `Dockerfile` and pushes `:latest` and `:<git-sha>`.

## 6. Deploy to Modal

```bash
set -a; . ./.env; set +a      # needs MODAL_TOKEN_ID, MODAL_TOKEN_SECRET,
                                # DOCKERHUB_USERNAME, DOCKERHUB_TOKEN set
pip install modal
bash deploy/modal_deploy.sh
```

This script, in order:
1. `modal token set` — authenticates the CLI with your account.
2. `modal secret create dockerhub-registry-secret ...` — lets Modal pull
   your **private** Docker Hub image (`Image.from_registry(..., secret=...)`).
3. `modal secret create gptoss-worker-secrets ...` — the worker's runtime
   config (model repo/file, llama-server tuning, `PROXY_API_KEY`,
   `LLAMA_INTERNAL_API_KEY`, CORS/rate-limit settings), stored encrypted in
   Modal, injected as env vars into the container.
4. `modal deploy modal_app.py` — builds the Modal Image on top of your
   registry image, and deploys the `Worker` class ([modal_app.py](modal_app.py)):
   - `min_containers=0`, `max_containers=1` → scale-to-zero, one GPU at a time
   - `gpu=["A100-80GB", "H100"]` → tries A100 80GB first, falls back to H100
   - `scaledown_window=60` → container shuts down 60s after the last request
   - `startup_timeout=1800` → generous budget for the very first cold start
     (download 61GB GGUF + load into VRAM); later cold starts are much
     faster since the volume already has the file

The deploy output prints the web endpoint URL, of the form:
```
https://<your-workspace>--gptoss-120b-worker-web.modal.run
```
That URL **is** your OpenAI-compatible base — append `/v1`.

Re-running `deploy/modal_deploy.sh` is safe: `modal secret create --force`
overwrites in place, and `modal deploy` is idempotent (updates the existing
app instead of creating a duplicate).

## 7. Test

**curl:**
```bash
curl https://<your-workspace>--gptoss-120b-worker-web.modal.run/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b-uncensored","messages":[{"role":"user","content":"Hello"}]}'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(api_key="<PROXY_API_KEY>", base_url="https://<your-workspace>--gptoss-120b-worker-web.modal.run/v1")

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

**Latency/throughput benchmark** (works against any OpenAI-compatible
`base_url`, Modal or Runpod):
```bash
pip install openai
PROXY_BASE_URL=https://<your-workspace>--gptoss-120b-worker-web.modal.run/v1 PROXY_API_KEY=... \
  python3 scripts/benchmark.py
```
Reports cold vs warm total latency, time-to-first-token, and tokens/sec.

## 8. Logs and scale-to-zero verification

```bash
python3 -m modal app logs gptoss-120b
```
Shows container boot (model download/skip, llama-server startup), per-request
timing (TTFT, generation latency) from `src/modal_gateway.py` and
`src/llama_runtime.py`'s logging. The Modal dashboard (`modal.com/apps`) also
shows live container count — confirm it's 0 while idle, goes to 1 on a
request, and back to 0 `scaledown_window` seconds after the last one.

## 9. Updating the model

- Same file, new weights: delete the file under `/cache/models/...` on the
  volume (`modal volume rm gptoss-120b-model-cache <path>` or
  `modal volume ls`/`get`/`rm`) and the next cold start re-downloads it.
- Different model/repo entirely: update `MODEL_REPO`/`MODEL_FILE`/
  `SERVED_MODEL_NAME` via `deploy/modal_deploy.sh` (it recreates the secret
  with `--force`) and redeploy, making sure the new file is GGUF-compatible
  with llama-server.

## 10. Tearing it down

```bash
python3 -m modal app stop gptoss-120b
python3 -m modal volume delete gptoss-120b-model-cache
python3 -m modal secret delete gptoss-worker-secrets
python3 -m modal secret delete dockerhub-registry-secret
```

## 11. Cost summary

| Resource | Billed when idle? |
|---|---|
| GPU container (`min_containers=0`) | **No** — $0 when no job is running |
| Modal Volume (model storage) | No separate line item in Modal's pricing model |
| Docker Hub / GitHub Actions | Free tier is enough for one private-ish image at this scale |
| Modal account | $30/month free credit on the Starter plan; pay-as-you-go beyond that, no prepaid minimum |

## 12. Known limitations / what to double-check before production

- `llama-server` CLI flags (`--n-cpu-moe`, `--flash-attn`, etc.) can change
  between llama.cpp releases — the Dockerfile builds `master` by default so
  the first build succeeds; pin `LLAMA_CPP_REF` to a tested release tag
  afterwards and re-verify flags with `llama-server --help` in that build.
  See the comment above `ARG LLAMA_CPP_REF` in [Dockerfile](Dockerfile).
- Modal's exact GPU identifier strings (`A100-80GB`, `H100`, etc.) and
  per-container concurrency behavior can change between SDK versions — this
  was verified against `modal==1.6.4`'s actual source
  (`Image.from_registry`, `@modal.web_server`/`@modal.asgi_app`,
  `@app.cls`/`@modal.enter`/`@modal.concurrent` signatures read directly from
  the installed package, not just docs) but not yet against a real deploy,
  which needs the Modal API token only you can provide.
- **Current status:** the worker image is built, pushed, and was verified
  end-to-end against local mocks (proxy/gateway translation, SSE streaming,
  auth, non-stream path). It has **not yet been run against the real Modal
  endpoint or the real 61GB model** — that needs `MODAL_TOKEN_ID` /
  `MODAL_TOKEN_SECRET` (see §4 and §6).
- Once deployed, re-run the cold/warm/scale-to-zero verification and the
  benchmark (§7-8) and update this section with real numbers.

---

## Appendix A: Runpod Serverless (alternative/legacy path)

This is the original deployment target. The worker image, model, and
llama.cpp runtime are identical to the Modal path above — only the
orchestration and proxy differ. Use this if you'd rather use Runpod and can
fund the account (Runpod requires **≥ $5 prepaid balance** to create a
network volume, which blocked this path for the original deployment).

### A.1 Architecture

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

### A.2 Prerequisites

- A **Runpod account + API key** (Settings → API Keys in the Runpod console)
  **with at least $5 balance** (required to create a network volume).
- A **Docker Hub access token with Read & Write scope** (Read-only is not
  enough to `docker push`).

### A.3 Create the Runpod endpoint

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

This creates, in order: a container registry auth (needed if your image is
private), a Network Volume (persistent model storage), a Serverless Template
(your image + env vars), and a Serverless Endpoint with `workersMin=0`
(Active Workers = 0), `workersMax=1`, Flex workers, `idleTimeout=60`,
`flashboot=true`, attached to the volume. Re-running the script reuses
whatever it already created (tracked in `deploy/infra_state.json` — delete
that file to force fresh resources).

Copy the printed `RUNPOD_ENDPOINT_ID` into your proxy's environment.

### A.4 Pre-warm the model (recommended, optional)

```bash
# Spin up a cheap CPU/GPU Runpod Pod with the SAME network volume attached
# at /workspace (or any path), then inside that pod:
MODEL_DIR=/workspace/models/gpt-oss-120b-uncensored ./scripts/download_model.sh
# Terminate the pod once it finishes — you only pay for pod time, not storage you already had.
```
Make sure the path your Pod wrote to matches `MODEL_DIR` as seen by the
*serverless* worker (`/runpod-volume/models/gpt-oss-120b-uncensored` — the
mount point is `/runpod-volume` for serverless workers specifically).

### A.5 Deploy the proxy

```bash
cd proxy
cp ../.env.example .env   # fill in RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, PROXY_API_KEY
docker build -t gptoss-proxy .
docker run --env-file .env -p 8000:8000 gptoss-proxy
```
`PROXY_API_KEY` is the key *your* clients will use — unrelated to
`RUNPOD_API_KEY`, which never leaves this service. Put a real domain + TLS in
front of it before exposing it publicly.

### A.6 Scale-to-zero test and benchmark

```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
./scripts/test_endpoint.sh
```
Checks worker count = 0, sends a request, times the cold start, checks
worker count ≥ 1, sends a warm request, waits past `idleTimeout`, checks
worker count is back to 0, then confirms it auto-starts again.

### A.7 Tearing it down

```bash
curl -X DELETE https://rest.runpod.io/v1/endpoints/<endpoint_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/templates/<template_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/networkvolumes/<volume_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
```
IDs are in `deploy/infra_state.json`. Deleting the network volume is what
stops the ongoing storage charge.

### A.8 Status when this path was abandoned

The worker image built and pushed successfully (verified: CUDA
compile+link, Docker Hub push with a Read & Write token). Container registry
auth and the fix for the exact Runpod REST API paths (`/networkvolumes` not
`/network-volumes`, `dataCenterId` not `dataCenter`) are already applied in
`deploy/create_infra.py`. The only remaining blocker was the **$5 minimum
account balance** Runpod requires to create a network volume — if you fund
the account, `deploy/create_infra.py` should complete successfully as-is.
