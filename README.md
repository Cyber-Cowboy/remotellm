# Open-weight LLMs on Runpod Serverless — scale-to-zero, OpenAI-compatible, multi-model

Self-host any GGUF model from Hugging Face on Runpod Serverless GPUs that
scale from **0 → 1 → 0** automatically, with a local dashboard that lets you
run **several models side by side**, pick hardware with its real price
before you commit, and talk to any of them through one OpenAI-compatible
`base_url`.

Originally built around
[`HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive)
(a derestricted fine-tune of OpenAI's open-weight `gpt-oss-120b`, 117B total /
~5.1B active params, MoE, Apache-2.0, 61GB GGUF/MXFP4) — but the worker image
and the dashboard's "Add a model" flow work with any single-file GGUF repo.

## 1. Architecture

```
Client (OpenAI SDK, AnythingLLM, curl)
   │  POST http://localhost:8787/v1/chat/completions   {"model": "<served_name>", ...}
   ▼
dashboard/ (FastAPI)                 <- checks auth, rate-limits, restricts CORS,
   │  routes by the request's           then routes by "model" to the right
   │  "model" field                     Runpod endpoint
   ▼
Runpod Serverless Endpoint            <- Flex workers, workersMin=0 (Active Workers = 0)
   │  idle -> 0 GPU workers, no billing   — one endpoint+template PER model
   │  request arrives -> Runpod starts a worker automatically
   ▼
Worker container (Dockerfile, src/handler.py)
   │  on cold start: download that model's GGUF into the shared network
   │  volume if missing, then start llama-server once and keep it warm
   ▼
llama-server (llama.cpp, CUDA)  --  OpenAI-compatible /v1/chat/completions,
   serving the GGUF from /runpod-volume/models/<served_name>/... (shared
   volume, one subfolder per model — never re-downloaded on later cold
   starts, and adding/removing one model never touches another's files)
```

**Why llama.cpp instead of vLLM:** gpt-oss-120b (and many community
fine-tunes) are published only as GGUF/MXFP4 files. vLLM's GGUF loader does
not reliably support MoE models in MXFP4 — it's explicitly experimental for
that combination. `llama-server` has first-class, actively-maintained
support for GGUF and already exposes an OpenAI-compatible API with
streaming, so it's the serving runtime here regardless of which specific
model you point it at.

**Why a shared persistent volume instead of baking models into the image:**
these files run 1-100+ GB. Baking one into the Docker image would make every
cold start pull that much, and any image rebuild (e.g. a handler bug fix)
would force a full re-upload/re-download. A Runpod network volume is
written to once per model (first cold start ever) and reused by every
worker that starts afterwards — only the ~2-3GB container image needs to be
pulled per worker, and it's the same image for every model.

## 2. GPU choice

61GB+ of weights plus KV cache does not fit on anything smaller than an
80GB-class GPU without offloading MoE experts to CPU (`N_CPU_MOE`, supported
by `llama-server` and exposed as an env var here if you need to downgrade to
a 48GB GPU at the cost of throughput). The dashboard's "Add a model" flow
estimates this automatically from the real GGUF file size and recommends
the cheapest GPU tier that fits — see §8.

Indicative Runpod Serverless pricing (hand-checked against
[runpod.io/pricing](https://www.runpod.io/pricing) — verify current rates
before relying on these, they change):

| GPU | VRAM | $/hr |
|---|---|---|
| RTX A4000 | 16GB | $0.58 |
| L4 | 24GB | $0.75 |
| RTX A6000 / A40 | 48GB | $1.22 |
| L40S | 48GB | $1.89 |
| A100 80GB | 80GB | $2.99 |
| H100 | 80GB | $5.89 |
| RTX PRO 6000 Blackwell | 96GB | $3.69 |
| H200 | 141GB | $6.99 |
| B200 | 180GB | $10.79 |

At `workersMin=0`, you only pay for the seconds a worker is actually
running (cold start + generation + the `idleTimeout` tail), never for idle
GPU time with no worker. The network volume itself has its own small,
continuous storage cost regardless of whether a worker is running (see §9).

## 3. Repository layout

```
.
├── deploy.sh                    # one-time bootstrap: shared registry auth + network volume + first endpoint
├── Dockerfile                   # worker image: llama.cpp (CUDA) + src/*.py
├── requirements.txt             # worker Python deps
├── src/
│   ├── llama_runtime.py        # shared: download model once, launch+health-check llama-server
│   └── handler.py              # Runpod generator handler: forwards/streams requests
├── deploy/
│   └── create_infra.py         # creates the shared network volume + template + endpoint via API
├── proxy/                       # standalone single-model OpenAI-compatible gateway (deploy anywhere)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── dashboard/                    # local control panel: multi-model, GPU/cost picker, unified gateway (see §8)
│   ├── main.py
│   └── static/index.html
├── scripts/
│   ├── run_dashboard.sh         # launches dashboard/ on http://localhost:8787
│   ├── run_proxy_local.sh       # runs proxy/ locally for a single endpoint, no dashboard
│   ├── print_base_url.sh        # prints the dashboard's base_url
│   ├── download_model.sh        # optional pre-warm of the network volume
│   ├── test_endpoint.sh         # scale-to-zero end-to-end test (single endpoint)
│   └── benchmark.py             # TTFT / tokens-per-sec / cold vs warm, via OpenAI SDK
├── .github/workflows/docker-build.yml   # builds+pushes the worker image (CI has the disk a laptop may not)
├── .env.example
└── .gitignore
```

## 4. Prerequisites you need to provide

- A **Runpod account + API key** with **≥ $5 balance** (Settings → API
  Keys; Billing → add funds — required to create a network volume) and a
  Docker Hub **Read-only** token so Runpod can pull your (private) image.
- A **Docker Hub account** with a repo to push `gptoss-120b-worker` to, and
  a **Read & Write** token for the push itself (separate from the
  Read-only one Runpod needs to pull).
- Nothing from Hugging Face for public models (the default one is public,
  Apache-2.0, no gating). `HF_TOKEN` is only there for the future, if you
  ever add a gated model.

None of these are stored anywhere in this repository — they live in a
local `.env` (gitignored) and, at deploy time, in Runpod's own template
env (never baked into the image or committed to git).

## 5. Build and push the worker image

Already done for this deployment; see
[`docker-build.yml`](.github/workflows/docker-build.yml). To rebuild after
a code change, push to `main` (or run the workflow manually from the
Actions tab); it builds `Dockerfile` and pushes
`docker.io/<DOCKERHUB_USERNAME>/gptoss-120b-worker:latest` and `:<git-sha>`.
Every model the dashboard deploys uses this same image — only the env vars
(which model/file, GPU, context window) differ per model.

## 6. One-time bootstrap

```bash
set -a; . ./.env; set +a   # RUNPOD_API_KEY, DOCKERHUB_USERNAME, ...
bash deploy.sh
```

Creates, in order: a container registry auth (so Runpod can pull a private
image), a Network Volume (persistent model storage, shared by every model
you'll add later), a Serverless Template, and a first Serverless Endpoint
with `workersMin=0`, `workersMax=1`, Flex workers, `idleTimeout=60`,
`flashboot=true`, attached to the volume. Re-running it reuses whatever
already exists (tracked in `deploy/infra_state.json` — delete that file to
force fresh resources). After this, use the dashboard (§8) to add/start/stop
models — it reuses the volume this creates.

**Important API gotchas already fixed here, in case you're comparing
against older examples:**
- The REST v1 paths are `/networkvolumes` (not `/network-volumes`) and the
  field is `dataCenterId` (not `dataCenter`).
- Not every data center supports network volumes — Runpod returns the
  valid list in the 500 error if you pick a wrong one; `US-KS-2` does
  **not** support them, `US-CA-2` does (current default here).
- GPU type ID strings are exact enum values, not free text — e.g. it's
  `"NVIDIA H100 PCIe"` / `"NVIDIA H100 80GB HBM3"`, not `"NVIDIA H100 80GB
  PCIe"` (Runpod returns the full valid enum list on a 400 if you guess
  wrong).
- **The job API (`/run`, `/runsync`, `/status`, `/stream`, `/health`) is at
  `api.runpod.ai`** — not `api.runpod.io`, which is only the management
  REST API (`/v1/endpoints`, `/v1/networkvolumes`, etc.). Mixing these up
  gives a silent 404, not an auth error. `proxy/main.py`,
  `scripts/test_endpoint.sh`, and `dashboard/main.py` all use the correct
  host.
- A Runpod template name must be unique, and a **deleted endpoint's
  template stays "pending cleanup" (reserving its name) for up to ~90
  minutes** — a template named purely from the model's served name
  collides on the very next Stop → Start within that window ("Template
  name must be unique"). The dashboard suffixes every template/endpoint
  name with a timestamp specifically to avoid this.
- A freshly created endpoint shows one worker in `idle`/`ready` state via
  `/health` even before you send a request — this is Runpod's flashboot
  **standby** worker (not client-configurable; `workersStandby` isn't a
  field the REST API accepts, contrary to some older examples). The
  ~$0.01/hr `currentSpendPerHr` you'll see at that point lines up with the
  network volume's own storage fee, not GPU compute — the GPU itself is
  still $0 until a job actually runs on it.
- **Cheap/small GPU tiers can have zero capacity in a given data center**,
  and Runpod's API accepts the job anyway — it just sits `IN_QUEUE` forever
  with no worker ever provisioning (confirmed by hand on RTX A4000 *and*
  RTX A6000, in the same data center, on different days). There's no error
  to catch; the dashboard flags a model as possibly-stuck after several
  minutes with zero workers (see §8) rather than leaving it looking broken.
  If this happens, pick a bigger/more common tier (A100/H100 are reliably
  available) instead of assuming something's wrong with the deployment.

## 7. Tearing everything down

```bash
curl -X DELETE https://rest.runpod.io/v1/endpoints/<endpoint_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/templates/<template_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/networkvolumes/<volume_id> -H "Authorization: Bearer $RUNPOD_API_KEY"
```
IDs are in `deploy/infra_state.json` and `dashboard/models.json`. Deleting
the network volume is what stops the ongoing storage charge. The dashboard's
"Delete everything" button (§8) does all of this for every tracked model in
one click.

## 8. The dashboard — multi-model control panel

```bash
set -a; . ./.env; set +a
bash scripts/run_dashboard.sh
# open http://localhost:8787
```

Run **several models side by side**, each its own Runpod endpoint+template,
all sharing one network volume (one subfolder per model, so adding or
removing one never touches another's downloaded weights):

- **Runpod account:** live balance + `$/hr` spend, and a **"Delete
  everything"** button that tears down every tracked model's
  endpoint+template plus the shared volume — nothing left billing, not even
  storage.
- **Models currently spending money:** the default view shows only models
  that actually have a GPU running right now (not every model you've ever
  added) — check "Show stopped/idle models too" to see the rest and restart
  them.
- **Add a model:** paste a Hugging Face link (a direct `.gguf` file link,
  or just `org/repo` if it has exactly one GGUF) and hit **"Check model &
  recommend hardware"**. This fetches the file's real size and its native
  context length straight from the GGUF's own metadata via Hugging Face's
  API (`?expand[]=gguf` — GGUF-only repos usually don't publish a
  `config.json`, but the GGUF metadata is always there), then shows every
  Runpod GPU tier with its real `$/hr` price, pre-selecting the cheapest
  one that fits. Pick a different tier if you want, optionally override
  the context window, then **Deploy**.
- **Click any model row** to expand it. A model with no live endpoint shows
  a **GPU picker with price** and a **Start** button right there — starting
  (or restarting after Stop) a model always means choosing hardware
  explicitly, never silently reusing whatever was picked last time. A
  running/startable model instead shows its recommended **context window**
  and **max tokens**, plus its `base_url`, model name, API key, and
  ready-to-paste `curl`/Python snippets.
- Every model is reachable at the **same** `base_url`
  (`http://localhost:8787/v1`) — the dashboard runs a small
  OpenAI-compatible gateway that routes each request by its `"model"`
  field, so you never need a different URL per model. This is what to put
  into AnythingLLM or any OpenAI-compatible client's "Generic OpenAI"
  settings.
- **Stop** deletes that model's endpoint+template (file stays on the
  shared volume, so the next Start doesn't re-download); **Remove** does
  the same and also forgets the model entirely.

`gpuCount` is fixed at 1 — there's no multi-GPU tensor-split, so a model
whose estimated VRAM exceeds the biggest single GPU (~180GB, B200) genuinely
doesn't fit; the recommendation flags this rather than guessing.

It binds to `127.0.0.1` only and holds `RUNPOD_API_KEY` server-side (never
sent to the browser) — don't expose it beyond localhost without adding real
authentication first, since every Add/Start/Stop button here spends or
stops real money.

## 9. Updating a model

Through the dashboard: **Remove** it, then **Add** it again with a new
Hugging Face link — same served name reuses the existing subfolder on the
shared volume (so if it's genuinely the same file, nothing re-downloads);
a different file downloads fresh under that name.

Manually: delete the file from `/runpod-volume/models/<served_name>/...`
(a short-lived Pod with the volume attached) and the next cold start for
that model re-downloads it.

## 10. Cost summary

| Resource | Billed when idle? |
|---|---|
| GPU worker (`workersMin=0`) per model | **No** — $0 when no job is running |
| Shared network volume (all models' weights) | **Yes** — flat per-GB-month storage fee, billed continuously regardless of traffic |
| Docker Hub / GitHub Actions | Free tier is enough for one private-ish image at this scale |

## 11. Testing a single endpoint without the dashboard

**Scale-to-zero cycle test** (talks to the job API directly):
```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
./scripts/test_endpoint.sh
```
Checks worker count = 0, sends a request, times the cold start, checks
worker count ≥ 1, sends a warm request, waits past `idleTimeout`, checks
worker count is back to 0, then confirms it auto-starts again.

**Standalone proxy** (one endpoint, no dashboard):
```bash
bash scripts/run_proxy_local.sh   # :8000/v1
```

**Latency/throughput benchmark** (works against the dashboard's gateway or
the standalone proxy — anything OpenAI-compatible):
```bash
pip install openai
PROXY_BASE_URL=http://localhost:8787/v1 PROXY_API_KEY=... python3 scripts/benchmark.py
```
Reports cold vs warm total latency, time-to-first-token, and tokens/sec.

## 12. Logs

- Worker logs (model download, llama-server startup, per-request
  cold/warm, TTFT, generation timing) — Runpod console → your endpoint →
  Workers → select a worker → Logs.
- Gateway logs — stdout of the dashboard (or `scripts/run_proxy_local.sh`
  for the standalone path).

Prompts are **not** logged by default (`LOG_PROMPTS=false`); flip it to
`true` only if you need to debug content and understand the privacy
implications of doing so.

## 13. Known limitations / what to double-check before production

- `llama-server` CLI flags (`--n-cpu-moe`, `--flash-attn`, etc.) can change
  between llama.cpp releases — the Dockerfile builds `master` by default so
  the first build succeeds; pin `LLAMA_CPP_REF` to a tested release tag
  afterwards and re-verify flags with `llama-server --help` in that build.
  See the comment above `ARG LLAMA_CPP_REF` in [Dockerfile](Dockerfile).
- Runpod's exact REST field/path names (§6) were discovered empirically
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
- **Verified end-to-end against the real, funded Runpod account:**
  - First-ever cold start (download the GGUF + load into VRAM): tens of
    minutes depending on file size (~27 minutes for the 61GB default
    model); one-time only, the model persists on the network volume.
  - Later cold start (worker restart, model already on the volume): as
    fast as ~90-100s.
  - Warm request latency: ~0.5–5s depending on prompt/response size; raw
    generation throughput on the default model was ~175–195 tokens/sec on
    an H100/A100 (llama-server's own `predicted_per_second`).
  - Scale-to-zero confirmed: GPU billing drops to ~$0.01/hr (network-volume
    storage only, not GPU compute) within the configured `idleTimeout`
    after the last request.
  - Multiple independent models (different repos/architectures, e.g.
    gpt-oss-120b and a Qwen3 GGUF) deployed and run **concurrently** on
    separate endpoints sharing the one network volume, each reachable
    through the dashboard's unified gateway by its own `model` name, both
    verified with the real `openai` Python SDK (streaming and
    non-streaming).
