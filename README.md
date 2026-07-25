# coload

> **co-load your models**: run Ollama and vLLM on one GPU without the VRAM fights.

`coload` is a lightweight, single-box orchestrator that lets **multiple local
inference engines** (Ollama, vLLM, and anything else) share **one GPU** safely:

- Measures live VRAM before every model load
- Fits the requested engine/model into what's actually free (minus a safety buffer)
- Serializes loads so they can't race
- When something doesn't fit, **alerts you to decide what to evict** rather
  than silently oversubscribing the card

## The problem

Engines don't know about each other's VRAM use, and they allocate differently:

- **Ollama** loads on demand and unloads on a keep-alive timer, but when VRAM
  is short it silently spills to CPU and crawls.
- **vLLM** grabs a fixed VRAM slice at process startup and holds it until the
  process exits.

The moment two of them want the card at once you get OOMs, CPU-spill slowdowns,
or a manual dance of "stop this, start that." `coload` fills the gap: **single
box, mixed engines, live-VRAM-aware, lazy lifecycle, human-in-the-loop when the
card is full.**

## How it works

Every load goes through one serialized routine, guarded by a single load-mutex
so only one load is ever in flight:

1. **Measure** free VRAM live (via NVML) at that exact moment.
2. **Budget**: free VRAM minus the safety headroom (`buffer_pct` of the card,
   yours to set, default 10%).
3. **Fits?** Start the engine sized to that budget (for vLLM this sets
   `gpu_memory_utilization`) and wait until it's healthy.
4. **Doesn't fit?** Alert you with what's needed, what's free, and what's
   resident, so *you* decide what to evict. Nothing is ever auto-killed.

Two invariants make it safe:

1. **Measure immediately before allocating, inside the mutex**, so the free
   figure can't go stale under a concurrent load.
2. **The buffer** (`buffer_pct`, default **10%**, configurable) absorbs
   fragmentation, CUDA context growth, and small fluctuations from other
   processes during the load.

Other GPU users (a CV inference server, desktop apps, games) are not managed;
they simply show up in the measured `used_vram`, and a background **watchdog**
alerts you if anything pushes total usage over the safe line.

## Quickstart

```bash
git clone https://github.com/GuiFV/coload && cd coload
uv sync
uv run coload serve
```

That's it. The repo ships a working [`config.yaml`](config.yaml); no copying
or renaming:

- **Ollama models** work if the Ollama daemon is running (default port
  `11434`). The defaults reference `llama3.2` and `nomic-embed-text`; swap in
  whatever `ollama list` shows on your box.
- **vLLM models** work if Docker is installed: coload brings the bundled
  [`docker-compose.yaml`](docker-compose.yaml) service up on first request and
  stops it again when idle.

Point any OpenAI-compatible client at the gateway:

```bash
curl http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]}'
```

`coload` routes by the `model` field, summons the right engine if it isn't
resident (scale-to-zero), and proxies the response, streaming included.

```bash
uv run coload status    # what's resident, VRAM map, budget
```

## Everyday CLI

```bash
coload models                      # list configured models + what's resident
coload up gemma4:12b               # spin a model up now (warm it)
coload chat gemma4:12b explain WSL in one line
coload down gemma4:12b             # unload it, freeing VRAM
coload status                      # VRAM map, budget, residency
```

All client commands take `--url`, or set `COLOAD_URL` once (useful from WSL
or another machine).

## Start at boot (Windows)

```bat
scripts\windows\install-autostart.cmd
```

Drops a launcher into your Startup folder that starts the gateway hidden at
every logon (no admin rights needed) and starts it immediately. Logs go to
`%LOCALAPPDATA%\coload\coload.log`. Remove with
`scripts\windows\uninstall-autostart.cmd`.

## Using from WSL

Install the CLI inside WSL from the mounted repo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install /mnt/c/path/to/co-load
```

Then point it at the Windows gateway. With default WSL networking (NAT) the
Windows host is the default gateway, so add to `~/.bashrc`:

```bash
export COLOAD_URL="http://$(ip route show default | awk '{print $3}'):8800"
```

Two networking notes:

- The gateway must bind `0.0.0.0` for NAT-mode WSL to reach it: set
  `host: "0.0.0.0"` in your `config.local.yaml`. That also exposes it on
  your LAN without auth; if you don't want that, keep the `127.0.0.1`
  default and enable WSL mirrored networking (`networkingMode=mirrored` in
  `%USERPROFILE%\.wslconfig`, then `wsl --shutdown`), after which
  `http://127.0.0.1:8800` works from WSL too.
- If Windows Defender Firewall prompts on first start, allow access on
  private networks.

## Configuration

The defaults in [`config.yaml`](config.yaml) run as-is. For your own setup,
copy it to `config.local.yaml` and edit there: it is gitignored and
`coload serve` prefers it automatically, so personal model lists and network
choices stay out of version control.

What to change, and why:

- **`engines.*.models`**: the routing table and the fit check's seed, so it
  must match reality. List the Ollama models you've actually pulled and the
  HF model ids you want vLLM to serve. A model not listed here gets a `404`.
- **`est_vram_gb`** per model: a rough figure is fine (model card size plus a
  little for KV cache). It only seeds the first fit decision; after the first
  real load coload measures actual usage and remembers it.
- **`buffer_pct`** (default `0.10`): raise it (`0.15`-`0.25`) on a desktop
  where browsers/games also use the GPU; lower it toward `0.05` on a headless
  box to squeeze in bigger models.
- **`idle_ttl_seconds`** (default `900`): lower frees VRAM sooner but means
  more cold starts; `0` disables idle-stop entirely.
- **`engines.vllm.start` / `stop`**: the default drives the bundled
  docker-compose service. Running vLLM bare-metal instead? Use the commented
  `vllm-bare` block: a non-detached `vllm serve` needs no `stop`, coload
  terminates the process itself.
- **`gpu`**: NVML index, for multi-GPU boxes where the arbitrated card isn't
  device `0`.
- **`host` / `port`**: move the gateway if `8800` clashes.

| Key | Default | Meaning |
| --- | --- | --- |
| `gpu` | `0` | NVML index of the GPU to arbitrate |
| `buffer_pct` | `0.10` | **User-settable.** Share of total VRAM always kept free as headroom. `[0, 1)` |
| `idle_ttl_seconds` | `900` | Stop a backend after this long idle (`0` disables) |
| `auto_evict_idle` | `false` | Opt-in: when a request doesn't fit, evict models **coload itself loaded** (LRU first) to make room before refusing. Out-of-band GPU users are still never evicted, only alerted about. For workloads that alternate between models too big to co-reside |
| `watchdog_interval_s` | `10` | Watchdog poll interval |
| `host` / `port` | `127.0.0.1` / `8800` | Gateway bind |
| `estimates_path` | `.coload/learned_estimates.json` | Where learned VRAM figures are cached |
| `alert.channels` | `[log]` | `log` and/or `webhook` (needs `alert.webhook_url`) |
| `engines.<name>.kind` | (required) | `ollama` or `vllm` |
| `engines.<name>.start` / `stop` | (vllm) | Command templates. `stop` is optional; set it when `start` detaches (e.g. `docker compose up -d` / `docker compose stop`) |
| `engines.<name>.models.<model>.est_vram_gb` | (required) | Seeds the fit check; refined from observed usage after the first real load |

## API surface

- `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`:
  OpenAI-compatible passthrough, admission-controlled.
- `GET /v1/models`: models configured across all engines.
- `POST /models/load`, `POST /models/unload` (body `{"model": "..."}`): warm a
  model up or evict it explicitly; what `coload up` / `coload down` call.
- `GET /status`: VRAM map, budget, what's resident per engine.
- Full card returns a clear `503` naming what's resident and how much you need
  to free.

## Known limitations

Stated plainly, because they're real:

- **Out-of-band loads.** If you load a model *directly* into Ollama (bypassing
  the gateway) while vLLM already holds VRAM, coload is not in that path. The
  watchdog **detects** the oversubscription and alerts, but cannot **prevent**
  it. The guarantee holds only for loads routed through the gateway; route all
  traffic through it.
- **Cold start.** Summoning a big model isn't free: a ~20 GB model takes
  ~30-90 s (weights to VRAM, CUDA init, warmup). Great for bursty workloads
  (batch pipelines); user-visible latency for the first interactive prompt
  after idle.
- **Single GPU (v1).** Multi-GPU placement is out of scope initially.
- **Estimates, not guarantees.** VRAM fit uses per-model estimates plus the
  buffer; pathological KV-cache growth on very long contexts can still
  surprise. The buffer + watchdog are the safety net.
- **Eviction is yours (v1).** When the card is full, the alert names what's
  resident; you decide what to stop. No auto-kill, by design.

## Development

```bash
uv sync
uv run pytest
```

The codebase is small and deliberately layered: `vram` (probe) ->
`estimates` / `alerts` -> `backends` (adapters) -> `orchestrator` (load-mutex
spine) -> `gateway` (admission point) -> `runtime` (composition root) -> `cli`.

## License

**coload is source-available, not OSI open source.**

- Public license: [PolyForm Small Business 1.0.0](LICENSE), **free for
  everyone**: individuals, research, education, nonprofits, government, and
  companies with fewer than 100 people **and** under 1,000,000 USD (2019,
  inflation-adjusted) annual revenue.
- **Commercial licensing:** companies over that threshold need a separate
  license. Contact <guilhermeviotti@gmail.com>.
- **Forks and derivatives** inherit these same terms automatically: a fork
  stays free for the same audience, and commercial use above the threshold
  still requires a license from the author.
