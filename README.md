# GPT-OSS-120B (uncensored) — scale-to-zero, OpenAI-compatible serverless

Self-hosted, OpenAI-SDK-compatible inference for
[`HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive)
(a derestricted fine-tune of OpenAI's open-weight `gpt-oss-120b`, 117B total /
~5.1B active params per token, MoE, Apache-2.0, distributed as a 61GB
GGUF/MXFP4 file), on GPU workers that scale from **0 → 1 → 0** automatically.

**Two interchangeable providers, one codebase.** The worker image, model,
and llama.cpp runtime are identical either way — only the cloud that
orchestrates scale-to-zero differs:

| | Runpod Serverless | Modal |
|---|---|---|
| Requires prepaid balance | **Yes**, ≥ $5 for a network volume | No, $30/mo free credit, but **requires a card on file** to use GPU functions |
| Gateway (auth/rate-limit/CORS) | Separate service (`proxy/`) you deploy yourself | Bundled into the same container (`src/modal_gateway.py`) |
| Client-facing endpoint | Wherever you deploy `proxy/` | One URL Modal gives you directly |

Pick whichever one you can actually fund right now with `PROVIDER=runpod`
or `PROVIDER=modal` in `.env` — see [§2 Switching providers](#2-switching-providers).

## 1. Architecture

```
Client (OpenAI SDK)
   │  POST <base_url>/v1/chat/completions
   ▼
OpenAI-compatible gateway           <- Runpod: proxy/main.py (standalone service)
   │  checks auth, rate-limits,       Modal:  src/modal_gateway.py (same container)
   │  restricts CORS
   ▼
Scale-to-zero GPU orchestrator       <- Runpod: Flex workers, workersMin=0
   │  idle -> 0 GPU workers,             Modal:  min_containers=0
   │  no billing
   │  request arrives -> 1 GPU container starts automatically
   ▼
Worker container (Dockerfile; src/handler.py or modal_app.py)
   │  on cold start: download the 61GB GGUF into persistent storage if
   │  missing, then start llama-server once and keep it warm
   ▼
llama-server (llama.cpp, CUDA)  --  OpenAI-compatible /v1/chat/completions,
   serving the GGUF from persistent storage (Runpod network volume /
   Modal Volume — never re-downloaded on later cold starts)
```

**Why llama.cpp instead of vLLM:** this specific model is published only as
a GGUF/MXFP4 file. vLLM's GGUF loader does not reliably support MoE models
in MXFP4 (gpt-oss architecture) — it's explicitly experimental for that
combination. `llama-server` has first-class, actively-maintained support for
exactly this format and already exposes an OpenAI-compatible API with
streaming, so it was chosen as the serving runtime per the project's
"if vLLM can't serve it, pick the most suitable runtime" rule.

**Why persistent storage instead of baking the model into the image:** the
model is 61GB. Baking it into the Docker image would make every cold start
pull 61GB, and any image rebuild (e.g. a handler bug fix) would force a full
model re-upload/re-download. A persistent volume is written to once (first
cold start ever) and reused by every container that starts afterwards —
only the ~2-3GB container image needs to be pulled per container.

## 2. Switching providers

Everything provider-specific is read from `.env`'s `PROVIDER` variable:

```bash
# .env
PROVIDER=runpod   # or: modal
```

```bash
set -a; . ./.env; set +a
bash deploy.sh                  # dispatches to the active provider
bash scripts/print_base_url.sh  # prints the OpenAI-compatible base_url to use
```

- **`PROVIDER=runpod`** → `deploy.sh` runs `deploy/create_infra.py` (network
  volume + template + endpoint), then tells you to run
  `scripts/run_proxy_local.sh` (or deploy `proxy/` somewhere real) to get an
  OpenAI-compatible URL in front of it.
- **`PROVIDER=modal`** → `deploy.sh` runs `deploy/modal_deploy.sh`, which
  does everything (secrets, image, deploy) and prints/saves the final URL to
  `deploy/modal_state.json`.

`scripts/benchmark.py` and the OpenAI SDK examples below take a plain
`base_url`/`PROXY_BASE_URL` — they don't know or care which provider is
behind it. One-off override without touching `.env`:
```bash
PROVIDER=modal bash deploy.sh
```

## 3. GPU choice

| | |
|---|---|
| Model weights (GGUF/MXFP4) | 61 GB on disk |
| MoE active params | ~5.1B per token (fast inference despite 117B total) |
| KV cache + runtime overhead | several more GB, scales with context length and concurrency |
| **Runpod GPU** | `RUNPOD_GPU_TYPE_IDS` — A100 80GB PCIe / H100 PCIe / A100-SXM4-80GB / H100 80GB HBM3 (priority order) |
| **Modal GPU** | `MODAL_GPU_TYPES` — A100-80GB, H100 (priority order) |

61GB of weights plus KV cache does not fit on anything smaller than an
80GB-class GPU without offloading MoE experts to CPU (`N_CPU_MOE`, supported
by `llama-server` and exposed as an env var here if you need to downgrade to
a 48GB GPU at the cost of throughput). 80GB GPUs were chosen over
multi-GPU tensor-parallel setups for simplicity (1 GPU per container).

Indicative pricing — verify current rates before relying on these, they
change ([Runpod](https://www.runpod.io/pricing) / [Modal](https://modal.com/pricing)):

| GPU | Runpod ~$/hr (flex) | Modal $/sec → $/hr |
|---|---|---|
| A100 80GB | ~$1.80–2.50/hr | $0.000694/sec → $2.50/hr |
| H100 80GB | ~$3.00–4.50/hr | $0.001097/sec → $3.95/hr |

Both providers only bill for the seconds a container is actually running
(cold start + generation + the idle tail) at `workersMin`/`min_containers`
= 0 — never for idle GPU time with no container running.

## 4. Repository layout

```
.
├── deploy.sh                    # dispatches to the active PROVIDER
├── Dockerfile                   # worker image: llama.cpp (CUDA) + src/*.py (shared by both providers)
├── requirements.txt             # worker Python deps (Runpod path; Modal installs its own at deploy time)
├── modal_app.py                  # Modal App definition (GPU, volume, secrets, autoscaling)
├── src/
│   ├── llama_runtime.py         # shared: download model once, launch+health-check llama-server
│   ├── modal_gateway.py         # FastAPI gateway used inside the Modal container (auth/rate-limit/CORS)
│   └── handler.py               # Runpod: generator handler, forwards/streams requests
├── deploy/
│   ├── modal_deploy.sh          # modal token/secret setup + `modal deploy` + saves modal_state.json
│   └── create_infra.py          # Runpod: creates network volume + template + endpoint via API
├── proxy/                       # Runpod: standalone OpenAI-compatible gateway (deploy anywhere)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── dashboard/                    # local control panel: balance/spend + Start/Stop (see §10.1)
│   ├── main.py
│   └── static/index.html
├── scripts/
│   ├── run_dashboard.sh         # launches dashboard/ on http://localhost:8787
│   ├── run_proxy_local.sh       # runs proxy/ locally with uvicorn (no Docker) for PROVIDER=runpod testing
│   ├── print_base_url.sh        # prints the right base_url for the active PROVIDER
│   ├── download_model.sh        # Runpod: optional pre-warm of the network volume
│   ├── test_endpoint.sh         # Runpod: scale-to-zero end-to-end test
│   └── benchmark.py             # TTFT / tokens-per-sec / cold vs warm, via OpenAI SDK (provider-agnostic)
├── .github/workflows/docker-build.yml   # builds+pushes the worker image (CI has the disk a laptop may not)
├── .env.example
└── .gitignore
```

## 5. Prerequisites you need to provide

- **Runpod:** account + API key with **≥ $5 balance** (Settings → API Keys;
  Billing → add funds), and a Docker Hub **Read-only** token so Runpod can
  pull your (private) image.
- **Modal:** account + API token (Settings → API Tokens), **and a payment
  method on file** — Modal requires this to use GPU functions even within
  the free-credit allowance; it's a verification requirement, not a charge.
- **Docker Hub:** an account with a repo to push `gptoss-120b-worker` to
  (already done — see §6), and a **Read & Write** token for the push itself
  (separate from the Read-only one Runpod needs to pull).
- Nothing is required from Hugging Face: the model is public, Apache-2.0,
  no gating. `HF_TOKEN` is only there for the future, if you ever swap in a
  gated model.

None of these are stored anywhere in this repository — they live in a local
`.env` (gitignored) and, at deploy time, in each provider's own encrypted
secret store (never baked into the image or committed to git).

## 6. Build and push the worker image

Shared by both providers — already done for this deployment; see
[`docker-build.yml`](.github/workflows/docker-build.yml). To rebuild after a
code change, push to `main` (or run the workflow manually from the Actions
tab); it builds `Dockerfile` and pushes
`docker.io/<DOCKERHUB_USERNAME>/gptoss-120b-worker:latest` and `:<git-sha>`.

## 7. Runpod deployment

```bash
set -a; . ./.env; set +a   # PROVIDER=runpod, RUNPOD_API_KEY, DOCKERHUB_USERNAME, ...
bash deploy.sh
```

This runs `deploy/create_infra.py`, which creates, in order: a container
registry auth (so Runpod can pull a private image), a Network Volume
(persistent model storage), a Serverless Template (your image + env vars),
and a Serverless Endpoint with `workersMin=0` (Active Workers = 0),
`workersMax=1`, Flex workers, `idleTimeout=60`, `flashboot=true`, attached
to the volume. Re-running it reuses whatever it already created (tracked in
`deploy/infra_state.json` — delete that file to force fresh resources).
Copy the printed `RUNPOD_ENDPOINT_ID` into `.env`.

**Important API gotchas already fixed here, in case you're comparing
against older examples:**
- The REST v1 paths are `/networkvolumes` (not `/network-volumes`) and the
  field is `dataCenterId` (not `dataCenter`).
- Not every data center supports network volumes — Runpod returns the valid
  list in the 500 error if you pick a wrong one; `US-KS-2` does **not**
  support them, `US-CA-2` does (current default here).
- GPU type ID strings are exact enum values, not free text — e.g. it's
  `"NVIDIA H100 PCIe"` / `"NVIDIA H100 80GB HBM3"`, not `"NVIDIA H100 80GB
  PCIe"` (Runpod returns the full valid enum list on a 400 if you guess wrong).
- **The job API (`/run`, `/runsync`, `/status`, `/stream`, `/health`) is at
  `api.runpod.ai`** — not `api.runpod.io`, which is only the management
  REST API (`/v1/endpoints`, `/v1/networkvolumes`, etc.). Mixing these up
  gives a silent 404, not an auth error. `proxy/main.py` and
  `scripts/test_endpoint.sh` both default to the correct host now.
- A freshly created endpoint shows one worker in `idle`/`ready` state via
  `/health` even before you send a request — this is Runpod's flashboot
  **standby** worker (not client-configurable; `workersStandby` isn't a
  field the REST API accepts, contrary to some older examples). The ~$0.01/hr
  `currentSpendPerHr` you'll see at that point lines up with the network
  volume's own storage fee, not GPU compute — the GPU itself is still $0
  until a job actually runs on it.

### 7.1 Deploy the gateway

For a quick local test, no Docker needed:
```bash
bash scripts/run_proxy_local.sh
```
For real use, deploy `proxy/` (a small stateless FastAPI app) wherever you
like — your own VM, Fly.io, Railway, a container platform:
```bash
cd proxy
docker build -t gptoss-proxy .
docker run --env-file ../.env -p 8000:8000 gptoss-proxy
```
`PROXY_API_KEY` is the key *your* clients use — unrelated to
`RUNPOD_API_KEY`, which never leaves this service. Put a real domain + TLS
in front of it before exposing it publicly, and keep `ALLOWED_ORIGINS`
scoped to your own frontend's origin (never `*`) if browsers call it
directly.

### 7.2 Pre-warm the model (recommended, optional)

`src/handler.py` downloads the 61GB GGUF automatically on its very first
cold start ever, but that first real user request would then wait through
the full download *and* model load (tens of minutes). To avoid that:
```bash
# Spin up a cheap CPU/GPU Runpod Pod with the SAME network volume attached
# at /workspace, then inside that pod:
MODEL_DIR=/workspace/models/gpt-oss-120b-uncensored ./scripts/download_model.sh
# Terminate the pod once it finishes — you only pay for pod time, not storage you already had.
```
Make sure the path matches `MODEL_DIR` as seen by the *serverless* worker
(`/runpod-volume/models/gpt-oss-120b-uncensored` — the mount point is
`/runpod-volume` for serverless workers specifically, which may differ from
the Pod's mount path).

### 7.3 Tearing it down

```bash
curl -X DELETE https://rest.runpod.io/v1/endpoints/<endpoint_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/templates/<template_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/networkvolumes/<volume_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
```
IDs are in `deploy/infra_state.json`. Deleting the network volume is what
stops the ongoing storage charge.

## 8. Modal deployment

```bash
set -a; . ./.env; set +a   # PROVIDER=modal, MODAL_TOKEN_ID/SECRET, DOCKERHUB_USERNAME, ...
pip install modal
bash deploy.sh
```

This runs `deploy/modal_deploy.sh`, which: authenticates the CLI
(`modal token set`), creates `dockerhub-registry-secret` (so Modal can pull
your private image) and `gptoss-worker-secrets` (runtime config, encrypted
in Modal), then `modal deploy modal_app.py`. [`modal_app.py`](modal_app.py)
defines a `Worker` class:
- `min_containers=0`, `max_containers=1` → scale-to-zero, one GPU at a time
- `gpu=["A100-80GB", "H100"]` → tries A100 80GB first, falls back to H100
- `scaledown_window=60` → container shuts down 60s after the last request
- `startup_timeout=1800` → budget for the very first cold start (download
  61GB + load into VRAM); later cold starts are much faster
- `@modal.enter()` downloads the model once and starts `llama-server`;
  `@modal.asgi_app()` serves `src/modal_gateway.py` (same container — no
  separate proxy service needed, unlike Runpod)

The script prints the endpoint URL and saves it to `deploy/modal_state.json`
(`scripts/print_base_url.sh` reads it back). Re-running is safe: secrets use
`--force`, and `modal deploy` updates the existing app in place.

**Known gotcha:** Modal requires a payment method on file to use GPU
functions at all, even inside the free-credit allowance — add one at
Settings → Billing before deploying, or `deploy.sh` will fail at the last
step with `Please add a payment method to use A100-80GB GPU functions.`

### 8.1 Tearing it down

```bash
python3 -m modal app stop gptoss-120b
python3 -m modal volume delete gptoss-120b-model-cache
python3 -m modal secret delete gptoss-worker-secrets
python3 -m modal secret delete dockerhub-registry-secret
```

## 9. Test

**curl:**
```bash
BASE_URL=$(bash scripts/print_base_url.sh 2>/dev/null)
curl "${BASE_URL}/chat/completions" \
  -H "Authorization: Bearer $PROXY_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b-uncensored","messages":[{"role":"user","content":"Hello"}]}'
```

**Python (OpenAI SDK) — identical for either provider:**
```python
from openai import OpenAI

client = OpenAI(api_key="<PROXY_API_KEY>", base_url="<BASE_URL from scripts/print_base_url.sh>")

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

**Runpod-specific scale-to-zero cycle test** (talks to the job API directly,
bypassing the gateway, to isolate Runpod/worker behavior):
```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
./scripts/test_endpoint.sh
```
Checks worker count = 0, sends a request, times the cold start, checks
worker count ≥ 1, sends a warm request, waits past `idleTimeout`, checks
worker count is back to 0, then confirms it auto-starts again. For Modal,
use the dashboard (`modal.com/apps`) or `modal app logs gptoss-120b` to
watch the same cycle.

**Latency/throughput benchmark (provider-agnostic):**
```bash
pip install openai
PROXY_BASE_URL=$(bash scripts/print_base_url.sh 2>/dev/null) PROXY_API_KEY=... \
  python3 scripts/benchmark.py
```
Reports cold vs warm total latency, time-to-first-token, and tokens/sec.

## 10. Logs

- **Runpod:** worker logs — Runpod console → your endpoint → Workers →
  select a worker → Logs. Gateway logs — stdout of wherever you deployed
  `proxy/` (or the terminal running `scripts/run_proxy_local.sh`).
- **Modal:** `python3 -m modal app logs gptoss-120b` shows both container
  boot (model download/skip, llama-server startup) and per-request gateway
  timing (TTFT, generation latency) in one stream, since they're the same
  container.

Prompts are **not** logged by default (`LOG_PROMPTS=false`) on either path;
flip it to `true` only if you need to debug content and understand the
privacy implications of doing so.

## 10.1 Control panel (multi-model, balance/spend, Start/Stop)

```bash
set -a; . ./.env; set +a
bash scripts/run_dashboard.sh
# open http://localhost:8787
```

A small local dashboard ([`dashboard/`](dashboard/)) for running **several
models side by side**, each its own Runpod endpoint or Modal app:

- **Provider accounts:** Runpod's live balance + `$/hr` spend, Modal's
  configured status, each with a **"Delete everything"** button (nukes
  every model on that provider plus the shared storage — see below).
- **Models currently spending money:** the default view shows only models
  that actually have a GPU running right now (not every model you've ever
  deployed) — check "Show stopped/idle models too" to see the rest and
  restart them. Each row has **Stop** (delete that model's endpoint/app;
  the file stays on the shared volume) and **Remove** (stop + forget it).
- **Add a model:** paste a Hugging Face link (a direct `.gguf` file link,
  or just `org/repo` if it has exactly one GGUF), pick a provider, and
  **"Check model & recommend hardware"** fetches the file's real size from
  HF, estimates the VRAM it needs, and shows every GPU tier your chosen
  provider offers with its real `$/hr` price — pre-selecting the cheapest
  one that fits. Pick a different tier if you want, then **Deploy**.
- **Click any model row** to expand its connection info: `base_url`,
  model name, API key, and a ready-to-paste `curl`/Python snippet. Every
  model, on either provider, is reachable at the **same** `base_url`
  (`http://localhost:8787/v1`) — the dashboard itself runs a small
  OpenAI-compatible gateway that routes each request by its `"model"`
  field, so you never need a different URL per model. This is what to put
  into AnythingLLM/any OpenAI-compatible client's "Generic OpenAI" settings.

All models on a provider share one network volume / Modal volume (each
model gets its own subfolder, named after its "served name"), so adding or
removing one model never re-downloads another's weights. `gpuCount` is
fixed at 1 everywhere — there's no multi-GPU tensor-split, so a model whose
estimated VRAM exceeds the biggest single GPU (~180GB, B200) genuinely
doesn't fit; the recommendation flags this rather than guessing.

**Known quirk, found while building this:** Runpod's cheaper/smaller GPU
tiers (tested: RTX A4000) can sit in `IN_QUEUE` indefinitely with zero
workers ever provisioning if that tier has no capacity in your data
center — the API accepts the request but nothing happens. If a model you
just added never starts, try a bigger/more common tier (A100/H100 are
reliably available) rather than assuming something's broken.

It binds to `127.0.0.1` only and holds `RUNPOD_API_KEY`/`MODAL_TOKEN_*`
server-side (never sent to the browser) — don't expose it beyond localhost
without adding real authentication first, since every Add/Start/Stop
button here spends or stops real money.

## 11. Updating the model

- Same file, new weights: delete it from persistent storage (Runpod: a
  short-lived Pod with the volume attached; Modal: `modal volume ls`/`get`/
  `rm` on `gptoss-120b-model-cache`) and the next cold start re-downloads it.
- Different model/repo entirely: update `MODEL_REPO`/`MODEL_FILE`/
  `SERVED_MODEL_NAME` in `.env` and re-run `bash deploy.sh` (Runpod: delete
  the relevant keys from `deploy/infra_state.json` first, or edit the
  template via the console/API; Modal: `deploy.sh` recreates the secret with
  `--force` automatically), making sure the new file is GGUF-compatible with
  llama-server.

## 12. Cost summary

| Resource | Runpod | Modal |
|---|---|---|
| GPU container, idle | **$0** at `workersMin`/`min_containers`=0 | **$0** at `min_containers=0` |
| Model storage | Network volume — flat per-GB-month fee, billed continuously regardless of traffic | Modal Volume — no separate line item in Modal's pricing |
| Free allowance | None | $30/month free credit (Starter plan) |
| Docker Hub / GitHub Actions | Free tier is enough for one private-ish image at this scale | (same) |

## 13. Known limitations / what to double-check before production

- `llama-server` CLI flags (`--n-cpu-moe`, `--flash-attn`, etc.) can change
  between llama.cpp releases — the Dockerfile builds `master` by default so
  the first build succeeds; pin `LLAMA_CPP_REF` to a tested release tag
  afterwards and re-verify flags with `llama-server --help` in that build.
  See the comment above `ARG LLAMA_CPP_REF` in [Dockerfile](Dockerfile).
- Modal's exact GPU identifier strings and per-container concurrency
  behavior were verified against `modal==1.6.4`'s actual installed source
  (`Image.from_registry`, `@modal.asgi_app`/`@modal.enter`/
  `@app.cls`/`@modal.concurrent` signatures read directly from the package,
  not just docs), but the real deploy was blocked on the payment-method
  requirement (§8) before it could finish — re-verify end-to-end if you pick
  this path.
- Runpod's exact REST field/path names (§7) were discovered empirically
  against the live API (several 400/500 responses that listed the valid
  enum/path), not solely from docs, which were stale or ambiguous on these
  specific points at the time this was built.
- **The Dockerfile had a real packaging bug, only catchable on a GPU host:**
  the CI build/push succeeding never meant the image actually *ran* —
  `llama-server` crash-looped on every real worker with "cannot open shared
  object file" because only the executable was copied out of the builder
  stage, not its sibling `.so` libraries, and even after fixing that,
  `/usr/local/bin` isn't on the dynamic linker's default search path and
  `libgomp1` wasn't installed. Both are fixed now (see git history), found
  by pulling the published image locally and running `ldd`/the real
  entrypoint directly — the CI build log alone couldn't have caught this.
- **Current status — verified end-to-end against the real, funded Runpod
  endpoint:**
  - First-ever cold start (download 61GB GGUF + load into VRAM): **~27
    minutes** (one-time only; the model persists on the network volume).
  - Later cold start (worker restart, model already on the volume): **~98s**.
  - Warm request latency: **~0.5–5s** depending on prompt/response size;
    raw generation throughput **~175–195 tokens/sec** on H100 (llama-server's
    own `predicted_per_second`).
  - Scale-to-zero confirmed: GPU billing drops to **$0.01/hr (network-volume
    storage only, not GPU compute)** within the configured `idleTimeout`
    after the last request; Runpod keeps one flashboot-cached "standby"
    worker visible as `idle` (not `running`) at all times by design — this
    is not literally zero *workers* in the API's own count, but it is zero
    GPU *cost*.
  - Tested through the full real chain with the actual `openai` Python SDK
    (both streaming and non-streaming) against the local gateway
    (`scripts/run_proxy_local.sh`) in front of the real endpoint — see
    `scripts/benchmark.py`'s output in the session for exact numbers.
  - The Modal path's code is complete and was verified as far as the Docker
    image build/push and secret creation; the actual GPU deploy step is
    blocked on Modal's payment-method requirement (§8) and hasn't been
    exercised against a real Modal container yet.
