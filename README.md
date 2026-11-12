# kai0 Inference

[English](README.md) | [简体中文](README_zh.md)

## BREAKING CHANGE: Forge 2.x / Dora 1.x migration

This migration from Forge 1.x / Dora 0.4.x requires coordinated deployment. Align the Forge packages, Dora CLI/coordinator/daemon, this node and its peer nodes before connecting a dataflow; replacing only one node in a Forge 1.x deployment is unsupported. Rebuild and redeploy executable artifacts with the matching environment. Keep `stable/1.x` for deployments that still use the previous stack.

The validated Python baseline is `forge-msgs==2.1.0`, `Pillow==13.0.0`, `forge-common==2.1.0`, the applicable `forge-robot` / `forge-policy` / `forge-kinematics` / `forge-tool` packages at `2.0.0`, and `dora-rs==1.0.1`. Respect this repository's Python and SDK platform requirements. Forge package families have independent versions. Rust nodes retain the published `forge_msgs` crate `2.0.0` with Dora `1.0.1`; Python Msgs `2.1.0` does not require a nonexistent Rust `2.1.0` release. The `forge_msgs.v1` message namespace remains unchanged.

This notice describes the deployment break from the previous stack; it does not declare hardware, real-model or sustained-run acceptance complete. See [this round's verification report](migration/msgs-2.1.0-report.json) for actual tests and outstanding gates.

> **Hardware deployment: [real_deploy.md](real_deploy.md)** covers installation, CAN/camera checks,
> mock acceptance, hardware startup, commands, expected outputs and troubleshooting.
> Use that guide as the authority for hardware deployment.

kai0 (χ₀) is an embodied manipulation policy based on openpi, with JAX/openpi weight loading and inference.
Unlike policies hosted directly in a LeRobot-style Node, its dependency stack conflicts with the Forge
policy environment. The Node therefore exchanges observations/actions with a separate inference server
over WebSocket, following openpi's deployment model. `kai0_runner` receives observations, calls the
server and forwards actions into Forge; `server/` runs JAX/openpi weight loading and inference in a
separate process.

## Node, Endpoint and Tool API

| Item | Current implementation |
| --- | --- |
| Production Nodes | 1 inference Node: `kai0 infer`; packaged entrypoint `kai0_policy`, see [build/make_node_bundle.sh](build/make_node_bundle.sh); `infer-once` is an offline CLI, not another Node |
| Tool support | Optional provider with inference `mode: session_endpoint`; default `dora_plain` registers no Endpoint |
| Endpoints | 1 in provider mode: `policy.runner`; 0 in plain mode |
| Operations | 1 in provider mode: `serve` (Session, stoppable, status supported, maximum concurrency 1) |
| Direct HTTP Tool API | 0; Dora `tool_in/tool_out` connect to Gateway |
| Gateway Tools | Defined by Skill/Gateway `tools.specs`; no repository-fixed external Tool ID |

Invocation mapping: `<tool_id from tools.specs> -> policy.runner / serve -> kai0 infer`.
The Session attaches to or launches a separate openpi inference service, runs readiness gates and
stops according to process ownership. The inference server is not another Dora Node.

Session uses stop, not Action cancel. Invoke through Gateway `POST /tools/{tool_id}:invoke`,
then read invocation status/results; stop with `POST /invocations/{invocation_id}/stop`.
Gateway owns these HTTP routes; the model inference service's interfaces are not Tool APIs.

Kai0 and LeRobot share Endpoint ID `policy.runner`. A Gateway has one current instance for this ID:
select them as deployment alternatives. Different Node IDs alone do not create separate routes.

Implementation and activation: [descriptor](inference/session_endpoint.py), [runner](inference/session_runner.py),
[configuration validation](inference/config.py), [inference entrypoint](inference/main.py).
Current Gateway source implements Session; historical binaries may differ, so check paired revisions and locks.

Compatibility: current dependency declarations are `forge-msgs>=2.0.0,<3`, `forge-common>=2.1.0,<3`, `forge-policy>=2.0.0,<3`, `forge-tool[dora]>=2.0.0,<3`, `dora-rs>=1.0.1,<2`. See [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). These declarations do not guarantee cross-version interoperability; also check the build revision and Skill lock when using artifacts.

Changes to Node entrypoints, Endpoints or Tool APIs must update this section and the associated configuration/contracts. Follow the [Node README contract](https://github.com/PhyAgentOS/forge/blob/master/CONTRIBUTING.md#node-readme-contract).

## System architecture

The system has three layers: server/client communicate over WebSocket, and client/environment communicate through Dora:

```text
Inference server (separate process, server/)
  serve_policy: JAX/openpi, checkpoint + norm_stats, WS port 8000
                         |
             WebSocket / msgpack binary frames
                         |
Client policy Node: kai0_runner
  observations -> remote inference -> actions; no local model loading
  config/inference/*.yaml; CLI or Dora Node
                         |
                 Dora 14D joint actions
                         |
Environment (choose one)
  Simulation: examples/dora_sim_kai0_dual_arm/ (local MJCF)
  Hardware: examples/dora_real_kai0_dual_arm/ (dual Piper arms + USB cameras)
```

**Deployment:** both inference server and client are required in either environment.
Choose simulation (`examples/dora_sim_kai0_dual_arm/`) for local demonstrations/validation or hardware
(`examples/dora_real_kai0_dual_arm/`) for physical deployment.

## Installation

Install once per machine; subsequent runs can use the deployment steps directly.
Client and server may share a machine or run separately. Keep the client (`uv` dependencies)
and server (JAX dependencies) environments separate.

Requirements: Linux x86_64, Python 3.12 and [uv](https://docs.astral.sh/uv/).
Client setup requires public Python indexes; `pyproject.toml` / `uv.lock` pin Forge packages without Git sources.
`server/setup.sh` also accesses github.com for LeRobot. The server requires an NVIDIA GPU with JAX CUDA;
historical measurements reported peak VRAM 12.8 GB on a 16 GB card and approximately 255 ms inference.
The lightweight WS client uses no GPU memory (dummy infer-once peak RSS approximately 44 MB).
Simulation and hardware environments both require the Forge workspace used in deployment step 3.

```bash
# Client Node, CLI and Dora examples; run from the repository root
uv sync --extra dev

# Inference server setup; first run needs network, UV_INDEX_URL can select a mirror
bash server/setup.sh
```

The main command is `kai0`, with `infer` / `infer-once` subcommands and compatibility commands
`kai0-infer` / `kai0-infer-once`. Unit tests are offline: `uv run pytest`.

## Deployment

> The full [hardware deployment guide](real_deploy.md) covers installation, CAN/camera checks,
> mock validation and physical startup, with commands and troubleshooting.
> The following sections explain mechanisms and provide parameter references.

### 1. Start the inference server

Deployment operators supply weights; see `ckpt/README.md` for placement.
Simulation and hardware use the same startup script, including action-space safety checks:

```bash
# --ckpt selects a step directory containing params/; its parent contains norm_stats.json
bash scripts/start_server.sh --ckpt 'ckpt/<task>/<step>' --rtc

# Replace task/step with the checkpoint directory provided for your deployment
bash scripts/start_server.sh --ckpt 'ckpt/<task>/<step>' --rtc
```

`--rtc` selects RTC mode (default; client `kai0_rtc.yaml`). Omit it for non-RTC operation
(client `kai0.yaml`, for debugging/fallback). Use `--delta` only with checkpoints trained
on relative joint angles; omit it for absolute-angle checkpoints.

First startup downloads the tokenizer into `OPENPI_DATA_HOME` (default `<repository>/cache/openpi`).
Offline deployments must prepopulate `cache/openpi/big_vision/` or export `OPENPI_DATA_HOME`
to an existing installation.

### 2. Warm up before starting the client

The first inference request compiles JAX (approximately 2.5 minutes in the documented fresh environment).
A warmup request brings it to steady state:

```bash
uv run kai0 infer-once --config config/inference/kai0_rtc.yaml
```

### 3. Start simulation or hardware

Both environments share the inference server, policy Node and Forge workspace. The environment dataflow
differs: a local MJCF simulator versus dual Piper arms and USB cameras. Both startup scripts also launch
the client embedded in their example dataflow; no separate client startup is required.

**Simulation** (`examples/dora_sim_kai0_dual_arm/`, RTC by default, stops after 90 seconds):

```bash
bash scripts/start_sim.sh 90
```

**Hardware** (`examples/dora_real_kai0_dual_arm/`, run after safety checks; use `--mock` without hardware first):

```bash
bash scripts/start_real.sh
```

Before hardware use, configure CAN `port:` in `piper_slave_left.yaml` / `piper_slave_right.yaml`
and camera `device:` in `camera_top/left/right.yaml` for the installation.
See the hardware example README's required pre-deployment parameters and confirmation commands.

## Inference configuration

| Configuration | Adapter | Purpose |
| --- | --- | --- |
| `config/inference/kai0_rtc.yaml` | `kai0_rtc` | RTC mode (default) |
| `config/inference/kai0.yaml` | `kai0` | Non-RTC debugging/fallback |
| `config/inference/kai0_libero.yaml` | `kai0_libero` | LIBERO benchmark: synchronous mode and LIBERO contract; see `examples/dora_bench_libero/` |
| `config/inference/dummy.yaml` | `dummy` | Offline debugging without a server |

See `config/inference/README.md` for fields and `inference/contract.py` for joint ordering and unit conversion.

## Repository layout

```text
.
├── cli.py                 # Unified CLI
├── real_deploy.md         # Installation, hardware checks, validation and startup
├── common/                # Shared path handling
├── inference/             # Client Dora and single-step inference
│   ├── contract.py        # Joint order, units, camera aliases: dual-arm / LIBERO profiles
│   ├── config.py          # Configuration loading and contract validation
│   ├── transport/         # Short/persistent WebSocket connections and msgpack codec
│   ├── policies/          # kai0, kai0_rtc, kai0_libero and dummy adapters
│   ├── main.py            # Dora Node
│   └── run_once.py        # infer-once
├── config/                # Templates; see config/inference/README.md
├── examples/              # See examples/README.md
│   ├── dora_sim_kai0_dual_arm/   # Local MJCF simulation + policy Node
│   ├── dora_real_kai0_dual_arm/  # Dual Piper arms + USB cameras + unit bridge
│   └── dora_bench_libero/        # Benchmark + kai0_libero policy
├── scripts/               # start_server.sh / start_sim.sh / start_real.sh
├── server/                # kai0_prompt_schedule/ snapshot + SOURCE.md
│   └── setup.sh           # Server environment setup
├── ckpt/                  # Weight placement; weights excluded, see ckpt/README.md
├── test/                  # Protocol-level fake_policy_server; see its README
└── tests/                 # Unit tests
```

## Forge 2.x migration

This branch prepares node version `2.0.0` for Dora 1.x, frozen to Dora `1.0.1` for migration validation. It uses the official Forge 2.x distributions; Common is at least `2.1.0`. Regenerated locks and all executable entrypoints are part of the change.

Every change report must list repository, changes and test evidence, including per-node `dora run` records and a chained test containing `adapter-forge-gateway`. Unavailable hardware/model checks remain pending. Source PRs do not publish stable releases or synchronize GitLab; synchronization follows full acceptance.
