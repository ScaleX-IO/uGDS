"""Unit tests for the dependency-free end-to-end benchmark client."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import bench_client


class FakeResponse:
    """Minimal streaming response returned by mocked urlopen calls."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return b"".join(self.lines)


class BenchClientTest(unittest.TestCase):
    def test_read_metrics_aggregates_engine_labels(self) -> None:
        metrics = b"""\
# HELP vllm:external_prefix_cache_hits External hits
vllm:external_prefix_cache_hits_total{engine=\"0\"} 128
vllm:external_prefix_cache_hits_total{engine=\"1\"} 256
vllm:external_prefix_cache_queries_total{engine=\"0\"} 512
vllm:prefix_cache_hits_total{engine=\"0\"} 0
"""
        with patch.object(bench_client, "http_get", return_value=metrics):
            values = bench_client.read_metrics("http://unused")

        self.assertEqual(
            bench_client.metric_value(
                values, "vllm:external_prefix_cache_hits"
            ),
            384,
        )
        self.assertEqual(
            bench_client.metric_value(
                values, "vllm:external_prefix_cache_queries"
            ),
            512,
        )

    def test_run_completion_parses_stream_usage(self) -> None:
        events = [
            {"choices": [{"text": "hello"}], "usage": None},
            {
                "choices": [],
                "usage": {"prompt_tokens": 1024, "completion_tokens": 1},
            },
        ]
        lines = [
            f"data: {json.dumps(event)}\n\n".encode("utf-8") for event in events
        ] + [b"data: [DONE]\n\n"]

        with patch(
            "bench_client.urllib.request.urlopen",
            return_value=FakeResponse(lines),
        ):
            result = bench_client.run_completion(
                "http://unused", "model", "prompt", 1, 1.0
            )

        self.assertEqual(result.prompt_tokens, 1024)
        self.assertEqual(result.completion_tokens, 1)
        self.assertGreaterEqual(result.latency_seconds, result.ttft_seconds)

    def test_read_metrics_requires_external_counter(self) -> None:
        with patch.object(
            bench_client,
            "http_get",
            return_value=b"vllm:prefix_cache_hits_total 0\n",
        ):
            with self.assertRaisesRegex(RuntimeError, "external prefix cache"):
                bench_client.read_metrics("http://unused")


if __name__ == "__main__":
    unittest.main()
