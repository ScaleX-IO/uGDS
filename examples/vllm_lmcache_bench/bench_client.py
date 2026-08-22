#!/usr/bin/env python3
"""Drive a cold/warm vLLM request pair and verify an external KV cache hit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


METRIC_NAMES = {
    "vllm:external_prefix_cache_hits",
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_queries_total",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_hits_total",
}


@dataclass
class RequestResult:
    """Timing and token counts from one streaming completion request."""

    ttft_seconds: float
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None


def http_get(url: str, timeout: float = 10.0) -> bytes:
    """Fetch a URL and return its response body."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed: HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"GET {url} failed: {error.reason}") from error


def read_metrics(base_url: str) -> dict[str, float]:
    """Read and aggregate the vLLM Prometheus counters used by this bench."""
    values = {name: 0.0 for name in METRIC_NAMES}
    body = http_get(f"{base_url}/metrics").decode("utf-8", errors="replace")
    found: set[str] = set()
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        if name not in values:
            continue
        try:
            values[name] += float(fields[1])
            found.add(name)
        except ValueError:
            continue
    if not any("external_prefix_cache_hits" in name for name in found):
        raise RuntimeError(
            "vLLM /metrics does not expose external prefix cache counters; "
            "make sure stats are enabled and this vLLM source is installed"
        )
    return values


def metric_value(metrics: dict[str, float], stem: str) -> float:
    """Return one counter regardless of Prometheus' optional _total suffix."""
    total_name = f"{stem}_total"
    if metrics.get(total_name, 0.0) != 0.0:
        return metrics[total_name]
    return metrics.get(stem, 0.0)


def make_prompt(repeats: int) -> str:
    """Build a deterministic, sufficiently long prefix for KV reuse."""
    paragraph = (
        "LMCache keeps reusable key value tensors outside the inference engine. "
        "uGDS moves those tensors directly between GPU memory and NVMe storage. "
        "This repeated benchmark passage is deterministic and intentionally long. "
    )
    return (
        "Read the following technical context and return one short summary.\n\n"
        + paragraph * repeats
        + "\nSummary:"
    )


def run_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> RequestResult:
    """Send one streaming completion and measure first-token and total latency."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    event_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                event_count += 1
                if first_token_at is None and event.get("choices"):
                    first_token_at = time.perf_counter()
                if event.get("usage"):
                    usage = event["usage"]
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"completion request failed: HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"completion request failed: {error.reason}") from error

    end = time.perf_counter()
    if event_count == 0 or first_token_at is None:
        raise RuntimeError("completion stream returned no token events")
    return RequestResult(
        ttft_seconds=first_token_at - start,
        latency_seconds=end - start,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def wait_for_metrics(
    base_url: str,
    initial_hits: float,
    timeout: float,
) -> dict[str, float]:
    """Wait for vLLM's periodic metric logger to publish the warm hit."""
    deadline = time.monotonic() + timeout
    latest: dict[str, float] = {}
    while time.monotonic() < deadline:
        latest = read_metrics(base_url)
        hits = metric_value(latest, "vllm:external_prefix_cache_hits")
        if hits > initial_hits:
            return latest
        time.sleep(1.0)
    return latest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-repeats", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--store-wait", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--metrics-timeout", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.prompt_repeats < 1:
        parser.error("--prompt-repeats must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    return args


def main() -> int:
    """Run the cold/warm benchmark and return nonzero unless uGDS is exercised."""
    args = parse_args()
    prompt = make_prompt(args.prompt_repeats)
    before = read_metrics(args.base_url)
    initial_external_hits = metric_value(
        before, "vllm:external_prefix_cache_hits"
    )
    initial_external_queries = metric_value(
        before, "vllm:external_prefix_cache_queries"
    )
    initial_local_hits = metric_value(before, "vllm:prefix_cache_hits")

    print("Running cold request...", flush=True)
    cold = run_completion(
        args.base_url, args.model, prompt, args.max_tokens, args.request_timeout
    )
    time.sleep(args.store_wait)

    print("Running warm request...", flush=True)
    warm = run_completion(
        args.base_url, args.model, prompt, args.max_tokens, args.request_timeout
    )
    after = wait_for_metrics(
        args.base_url, initial_external_hits, args.metrics_timeout
    )

    external_hits = (
        metric_value(after, "vllm:external_prefix_cache_hits")
        - initial_external_hits
    )
    external_queries = (
        metric_value(after, "vllm:external_prefix_cache_queries")
        - initial_external_queries
    )
    local_hits = metric_value(after, "vllm:prefix_cache_hits") - initial_local_hits
    passed = external_hits > 0 and external_queries > 0
    speedup = cold.ttft_seconds / warm.ttft_seconds if warm.ttft_seconds else None
    result = {
        "passed": passed,
        "model": args.model,
        "prompt_repeats": args.prompt_repeats,
        "cold": asdict(cold),
        "warm": asdict(warm),
        "ttft_speedup": speedup,
        "external_cache_query_tokens": external_queries,
        "external_cache_hit_tokens": external_hits,
        "local_prefix_cache_hit_tokens": local_hits,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(f"Cold TTFT: {cold.ttft_seconds:.4f} s")
    print(f"Warm TTFT: {warm.ttft_seconds:.4f} s")
    print(f"TTFT speedup: {speedup:.2f}x")
    print(f"External query tokens: {external_queries:.0f}")
    print(f"External hit tokens: {external_hits:.0f}")
    print(f"Local prefix-cache hit tokens: {local_hits:.0f}")
    if passed:
        print(
            "PASS: LMCache external cache hit increased by "
            f"{external_hits:.0f} tokens"
        )
        return 0
    print(
        "FAIL: no LMCache external cache hit was observed; inspect the service logs",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
