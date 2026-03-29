# LLM Inference Playground: SGLang vs vLLM

A side-by-side comparison tool for evaluating LLM inference performance using **SGLang** and **vLLM** backends. Send prompts to both engines simultaneously, compare outputs, and benchmark latency and throughput — all from a clean web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Why This Exists

Choosing between SGLang and vLLM for LLM serving is a common decision. This tool lets you:

- **Compare outputs** side-by-side for the same prompt
- **Benchmark latency** (time-to-first-token, total generation time)
- **Measure throughput** (tokens/sec) across both backends
- **Run batch benchmarks** with configurable parameters
- **Visualize results** with interactive charts

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Browser UI  │────▶│  FastAPI Backend  │────▶│   SGLang    │
│  (HTML/JS)   │◀────│  /api/*           │────▶│   vLLM      │
└─────────────┘     └──────────────────┘     └─────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start LLM Backends

**Option A: Use real SGLang/vLLM servers** (requires GPU)

```bash
# Terminal 1: Start SGLang
python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000

# Terminal 2: Start vLLM
python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**Option B: Use demo mode** (no GPU needed — uses mock responses for testing the UI)

```bash
# Just start the app — it auto-detects missing backends and enables demo mode
python3 app.py --demo
```

### 3. Launch the Playground

```bash
python3 app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `SGLANG_URL` | `http://localhost:30000` | SGLang server URL |
| `VLLM_URL` | `http://localhost:8000` | vLLM server URL |
| `PORT` | `7860` | Playground web UI port |

## Features

- **Prompt Playground**: Send a prompt to both backends, see outputs and timing side-by-side
- **Batch Benchmark**: Run N prompts with configurable `max_tokens` and `temperature`, get aggregate stats
- **Live Charts**: Visualize latency distribution and throughput comparison
- **Prompt Templates**: Pre-built prompts for common tasks (summarization, code gen, Q&A)
- **Export Results**: Download benchmark results as JSON

## Demo Mode

If you don't have GPU access, `--demo` mode simulates both backends with realistic latency profiles so you can explore the full UI and workflow.

## Tech Stack

- **Backend**: Python, FastAPI, httpx (async HTTP)
- **Frontend**: Vanilla HTML/CSS/JS, Chart.js
- **LLM Engines**: SGLang, vLLM (OpenAI-compatible API)

## License

MIT
