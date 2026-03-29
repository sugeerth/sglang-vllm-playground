"""LLM Inference Playground: SGLang vs vLLM comparison tool."""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="LLM Inference Playground")

SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:30000")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
DEMO_MODE = False


# --- Models ---

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class BenchmarkRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = 256
    temperature: float = 0.7


# --- Demo mode mock ---

DEMO_RESPONSES = [
    "Large language models have revolutionized natural language processing by enabling machines to understand and generate human-like text. They work by predicting the next token in a sequence based on the context provided by previous tokens.",
    "The key difference between SGLang and vLLM lies in their optimization strategies. SGLang focuses on structured generation and RadixAttention for prefix caching, while vLLM pioneered PagedAttention for efficient memory management during inference.",
    "To optimize LLM inference, consider these approaches: 1) Use KV-cache efficiently, 2) Batch requests to maximize GPU utilization, 3) Choose appropriate quantization (AWQ, GPTQ), and 4) Use speculative decoding for latency-sensitive applications.",
    "Python's asyncio library provides a foundation for writing concurrent code using the async/await syntax. It's particularly useful for I/O-bound operations like making HTTP requests to multiple LLM serving endpoints simultaneously.",
    "Transfer learning allows models pre-trained on large datasets to be fine-tuned for specific tasks with relatively small amounts of task-specific data. This has made state-of-the-art NLP accessible even with limited computational resources.",
]


async def mock_inference(backend: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    """Simulate inference with realistic latency profiles."""
    # SGLang is generally faster due to RadixAttention prefix caching
    if backend == "sglang":
        base_latency = random.uniform(0.3, 0.8)
        tokens_per_sec = random.uniform(45, 65)
    else:
        base_latency = random.uniform(0.4, 1.0)
        tokens_per_sec = random.uniform(35, 55)

    response_text = random.choice(DEMO_RESPONSES)
    # Truncate to approximate max_tokens (rough: 1 token ~ 4 chars)
    response_text = response_text[: max_tokens * 4]
    token_count = max(1, len(response_text) // 4)
    gen_time = token_count / tokens_per_sec

    await asyncio.sleep(base_latency + gen_time)

    total_time = base_latency + gen_time
    return {
        "text": response_text,
        "tokens_generated": token_count,
        "latency_ms": round(total_time * 1000, 1),
        "tokens_per_sec": round(token_count / total_time, 1),
        "time_to_first_token_ms": round(base_latency * 1000, 1),
    }


# --- Real backend calls ---

async def call_backend(client: httpx.AsyncClient, base_url: str, backend: str,
                       prompt: str, max_tokens: int, temperature: float) -> dict:
    """Call an OpenAI-compatible endpoint and return timing stats."""
    if DEMO_MODE:
        return await mock_inference(backend, prompt, max_tokens, temperature)

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", max(1, len(text) // 4))

        return {
            "text": text,
            "tokens_generated": completion_tokens,
            "latency_ms": round(elapsed * 1000, 1),
            "tokens_per_sec": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0,
            "time_to_first_token_ms": round(elapsed * 200, 1),  # approximate
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "text": f"[Error: {e}]",
            "tokens_generated": 0,
            "latency_ms": round(elapsed * 1000, 1),
            "tokens_per_sec": 0,
            "time_to_first_token_ms": 0,
            "error": str(e),
        }


# --- API Routes ---

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
async def status():
    """Check backend availability."""
    if DEMO_MODE:
        return {"sglang": "demo", "vllm": "demo", "demo_mode": True}

    results = {}
    async with httpx.AsyncClient() as client:
        for name, url in [("sglang", SGLANG_URL), ("vllm", VLLM_URL)]:
            try:
                r = await client.get(f"{url}/v1/models", timeout=5)
                results[name] = "online" if r.status_code == 200 else "error"
            except Exception:
                results[name] = "offline"
    results["demo_mode"] = False
    return results


@app.post("/api/infer")
async def infer(req: InferenceRequest):
    """Run inference on both backends and return comparison."""
    async with httpx.AsyncClient() as client:
        sglang_task = call_backend(client, SGLANG_URL, "sglang",
                                   req.prompt, req.max_tokens, req.temperature)
        vllm_task = call_backend(client, VLLM_URL, "vllm",
                                 req.prompt, req.max_tokens, req.temperature)
        sglang_result, vllm_result = await asyncio.gather(sglang_task, vllm_task)

    return {
        "sglang": sglang_result,
        "vllm": vllm_result,
    }


@app.post("/api/benchmark")
async def benchmark(req: BenchmarkRequest):
    """Run batch benchmark across both backends."""
    all_sglang = []
    all_vllm = []

    async with httpx.AsyncClient() as client:
        for prompt in req.prompts:
            s_task = call_backend(client, SGLANG_URL, "sglang",
                                  prompt, req.max_tokens, req.temperature)
            v_task = call_backend(client, VLLM_URL, "vllm",
                                  prompt, req.max_tokens, req.temperature)
            s_res, v_res = await asyncio.gather(s_task, v_task)
            all_sglang.append(s_res)
            all_vllm.append(v_res)

    def aggregate(results):
        latencies = [r["latency_ms"] for r in results]
        tps = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
        return {
            "results": results,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1) if latencies else 0,
            "avg_tokens_per_sec": round(sum(tps) / len(tps), 1) if tps else 0,
            "total_requests": len(results),
        }

    return {
        "sglang": aggregate(all_sglang),
        "vllm": aggregate(all_vllm),
    }


# Mount static files (after routes so / is handled by the route)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Playground")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with mock backends")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    args = parser.parse_args()

    DEMO_MODE = args.demo
    if DEMO_MODE:
        print("Running in DEMO mode (mock backends)")

    print(f"Starting LLM Inference Playground on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
