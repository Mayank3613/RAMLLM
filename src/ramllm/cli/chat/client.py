"""OpenAI-compatible streaming chat client for RAMLLM interactive REPL."""

import json
import time
from collections.abc import Generator
from typing import Any

import httpx
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat turn in conversation history."""

    role: str  # 'system', 'user', or 'assistant'
    content: str
    timestamp: float = Field(default_factory=time.time)
    interrupted: bool = False


class ChatMetrics(BaseModel):
    """Performance and latency metrics for a generation turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_s: float = 0.0
    total_generation_time_s: float = 0.0
    tokens_per_second: float = 0.0


class SessionStats(BaseModel):
    """Cumulative metrics across the entire chat REPL session."""

    total_turns: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_duration_s: float = 0.0
    last_metrics: ChatMetrics | None = None

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_duration_s > 0 and self.total_completion_tokens > 0:
            return round(self.total_completion_tokens / self.total_duration_s, 1)
        return 0.0


class ChatClient:
    """Client for streaming completions from OpenAI-compatible llama.cpp or RAMLLM servers."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 120.0):
        # Normalize base URL (strip trailing slash and ensure /v1 accessibility)
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.api_url = f"{self.base_url}/v1"
        else:
            self.api_url = self.base_url
        self.timeout = timeout
        self.stats = SessionStats()
        self.last_metrics: ChatMetrics | None = None

    def check_health(self) -> bool:
        """Check if the backend engine is running and responding."""
        endpoints = [
            f"{self.base_url}/health",
            f"{self.base_url}/healthz",
            f"{self.api_url}/models",
        ]
        with httpx.Client(timeout=2.0) as client:
            for ep in endpoints:
                try:
                    res = client.get(ep)
                    if res.status_code in (200, 204):
                        return True
                except Exception:
                    continue
        return False

    def estimate_prompt_tokens(self, messages: list[ChatMessage]) -> int:
        """Rough token count estimation (~4 characters per token)."""
        total_chars = sum(len(m.content) for m in messages)
        return max(1, total_chars // 4)

    def stream_chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Stream chat completions from backend via Server-Sent Events (SSE).

        Yields text chunks as they arrive from the backend.
        Updates self.last_metrics and self.stats once stream completes or is stopped.
        """
        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.api_url}/chat/completions"
        t_start = time.perf_counter()
        t_first: float | None = None
        chunk_count = 0
        total_chars = 0
        reported_prompt_tokens: int | None = None
        reported_completion_tokens: int | None = None

        with httpx.Client(timeout=self.timeout) as client:
            try:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = response.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"API error ({response.status_code}): {err_text}")

                    for raw_line in response.iter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Extract delta content
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if t_first is None:
                                    t_first = time.perf_counter()
                                chunk_count += 1
                                total_chars += len(content)
                                yield content

                        # Extract usage info if backend supplies it
                        usage = chunk.get("usage")
                        if usage:
                            reported_prompt_tokens = usage.get("prompt_tokens")
                            reported_completion_tokens = usage.get("completion_tokens")
            finally:
                t_end = time.perf_counter()
                gen_time = max(0.001, t_end - (t_first or t_start))
                first_latency = (t_first - t_start) if t_first is not None else (t_end - t_start)

                c_tokens = reported_completion_tokens or chunk_count or (total_chars // 4)
                p_tokens = reported_prompt_tokens or self.estimate_prompt_tokens(messages)
                tok_s = round(c_tokens / gen_time, 1) if (c_tokens and gen_time > 0) else 0.0

                metrics = ChatMetrics(
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens,
                    time_to_first_token_s=round(first_latency, 3),
                    total_generation_time_s=round(t_end - t_start, 3),
                    tokens_per_second=tok_s,
                )
                self.last_metrics = metrics
                self.stats.total_turns += 1
                self.stats.total_prompt_tokens += p_tokens
                self.stats.total_completion_tokens += c_tokens
                self.stats.total_duration_s += gen_time
                self.stats.last_metrics = metrics

    def complete_chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, ChatMetrics]:
        """Perform a non-streaming completion turn (e.g. for piped or one-shot prompt execution)."""
        tokens: list[str] = []
        for chunk in self.stream_chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
        ):
            tokens.append(chunk)

        content = "".join(tokens)
        metrics = self.last_metrics or ChatMetrics()
        return content, metrics
