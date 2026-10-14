"""Unit and integration tests for RAMLLM interactive chat REPL (Milestone 5)."""

import http.server
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.document import Document
from tests.ux.test_harness import assert_clean_process_state

from ramllm.backends.mock import MockBackendAdapter
from ramllm.cli.chat.client import ChatClient, ChatMessage, ChatMetrics
from ramllm.cli.chat.history import get_chat_history
from ramllm.cli.chat.render import ChatRenderer
from ramllm.cli.chat.repl import ChatSession
from ramllm.cli.chat.slash import SlashCommandCompleter, handle_slash_command
from ramllm.cli.chat.toolbar import build_bottom_toolbar, format_tokens
from ramllm.cli.chat.transcripts import load_transcript, save_transcript
from ramllm.core.schemas import (
    ExecutionPlan,
    KvConfig,
    MemoryEstimate,
    Placement,
    SpeedEstimate,
)


class MockOpenAIServer(http.server.BaseHTTPRequestHandler):
    """Minimal mock HTTP server responding to OpenAI-compatible SSE chat completions."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Quiet logger

    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz", "/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "models": []}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path.endswith("/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            stream = body.get("stream", False)

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                # Stream out chunks
                tokens = ["Hello", " from", " RAMLLM", " mock", " backend!"]
                for tok in tokens:
                    chunk = {
                        "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.01)

                final_chunk = {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {
                    "choices": [
                        {"message": {"role": "assistant", "content": "Hello from non-stream!"}}
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif self.path.endswith("/ramllm/replan"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "replanned"}')
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def mock_server():
    """Start mock OpenAI SSE server on a random free localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), MockOpenAIServer)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def make_dummy_plan() -> ExecutionPlan:
    """Construct minimal valid ExecutionPlan for chat session tests."""
    return ExecutionPlan(
        id="balanced-gpu-test",
        model_variant="test-7b",
        backend="mock",
        placement=Placement(gpu_layers={0: list(range(32))}, cpu_layers=[]),
        kv=KvConfig(location="gpu", type_k="q8_0", type_v="q8_0", context=4096),
        memory=MemoryEstimate(
            vram_peak_bytes={0: 5 * 1024**3},
            ram_peak_bytes=2 * 1024**3,
        ),
        speed=SpeedEstimate(
            decode_tok_s_low=25.0,
            decode_tok_s_high=35.0,
            prefill_tok_s_low=100.0,
            prefill_tok_s_high=150.0,
            bottleneck="vram_bw",
        ),
        verdict="COMFORTABLE",
        warnings=[],
        backend_args=["--ctx", "4096"],
        explanation=["Fast GPU-resident execution."],
    )


# =====================================================================
# 1. CLIENT TESTS
# =====================================================================


def test_client_health_and_streaming(mock_server: str) -> None:
    client = ChatClient(base_url=mock_server)
    assert client.check_health() is True

    messages = [ChatMessage(role="user", content="Hello!")]
    tokens: list[str] = []
    for chunk in client.stream_chat(messages):
        tokens.append(chunk)

    full_text = "".join(tokens)
    assert full_text == "Hello from RAMLLM mock backend!"

    metrics = client.last_metrics
    assert metrics is not None
    assert metrics.completion_tokens == 5
    assert metrics.prompt_tokens == 8
    assert metrics.tokens_per_second > 0
    assert client.stats.total_turns == 1
    assert client.stats.total_completion_tokens == 5


def test_client_complete_chat(mock_server: str) -> None:
    client = ChatClient(base_url=mock_server)
    messages = [ChatMessage(role="user", content="Test non-stream")]
    content, metrics = client.complete_chat(messages)
    assert content == "Hello from RAMLLM mock backend!"
    assert metrics.completion_tokens == 5


# =====================================================================
# 2. TRANSCRIPT TESTS
# =====================================================================


def test_save_and_load_markdown_transcript(tmp_path: Path) -> None:
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is RAMLLM?"),
        ChatMessage(
            role="assistant", content="RAMLLM is an adaptive inference manager.", interrupted=False
        ),
        ChatMessage(role="user", content="Tell me more"),
        ChatMessage(role="assistant", content="It optimizes RAM and GPU VRAM.", interrupted=True),
    ]
    out_file = tmp_path / "test_chat.md"
    saved = save_transcript(
        messages, model_name="qwen-2.5-7b", plan_id="plan-1", dest_path=out_file
    )
    assert saved.exists()

    loaded = load_transcript(saved)
    assert len(loaded) == 5
    assert loaded[0].role == "system"
    assert loaded[0].content == "You are a helpful assistant."
    assert loaded[1].role == "user"
    assert loaded[1].content == "What is RAMLLM?"
    assert loaded[2].role == "assistant"
    assert loaded[2].content == "RAMLLM is an adaptive inference manager."
    assert loaded[4].role == "assistant"
    assert loaded[4].content == "It optimizes RAM and GPU VRAM."


def test_save_and_load_json_transcript(tmp_path: Path) -> None:
    messages = [
        ChatMessage(role="user", content="Ping"),
        ChatMessage(role="assistant", content="Pong"),
    ]
    out_file = tmp_path / "transcript.json"
    saved = save_transcript(messages, model_name="test-model", dest_path=out_file)
    assert saved.exists()

    loaded = load_transcript(saved)
    assert len(loaded) == 2
    assert loaded[0].content == "Ping"
    assert loaded[1].content == "Pong"


# =====================================================================
# 3. TOOLBAR TESTS
# =====================================================================


def test_toolbar_formatting(mock_server: str) -> None:
    assert format_tokens(500) == "500"
    assert format_tokens(4096) == "4.1k"
    assert format_tokens(8192) == "8.2k"

    client = ChatClient(base_url=mock_server)
    plan = make_dummy_plan()
    session = ChatSession(
        model_name="Qwen2.5-7B-Instruct",
        client=client,
        plan=plan,
        toolbar_enabled=True,
    )
    session.client.last_metrics = ChatMetrics(tokens_per_second=32.4)

    toolbar = build_bottom_toolbar(session)
    assert "Qwen2.5-7B-Instruct" in toolbar
    assert "ctx" in toolbar
    assert "32.4 tok/s" in toolbar

    session.toolbar_enabled = False
    assert build_bottom_toolbar(session) == ""


# =====================================================================
# 4. SLASH COMMAND & COMPLETER TESTS
# =====================================================================


def test_slash_completer() -> None:
    completer = SlashCommandCompleter()

    # Match "/he" -> "/help"
    completions = list(completer.get_completions(Document("/he"), None))  # type: ignore[arg-type]
    names = [c.text for c in completions]
    assert "/help" in names

    # Sub-completion "/mode "
    completions_mode = list(completer.get_completions(Document("/mode f"), None))  # type: ignore[arg-type]
    assert any(c.text == "fast" for c in completions_mode)

    # Sub-completion "/toolbar "
    completions_tb = list(completer.get_completions(Document("/toolbar o"), None))  # type: ignore[arg-type]
    tb_names = [c.text for c in completions_tb]
    assert "on" in tb_names or "off" in tb_names


def test_slash_commands_execution(mock_server: str, tmp_path: Path) -> None:
    client = ChatClient(base_url=mock_server)
    session = ChatSession(
        model_name="test-model",
        client=client,
        plan=make_dummy_plan(),
        is_attached=True,
    )

    # /help returns False (don't exit)
    assert handle_slash_command("/help", session) is False

    # /system updates system message
    assert handle_slash_command("/system You are a pirate", session) is False
    assert session.system_prompt == "You are a pirate"
    assert session.messages[0].content == "You are a pirate"

    # /temp, /top_p, /seed
    handle_slash_command("/temp 0.5", session)
    assert session.temperature == 0.5
    handle_slash_command("/top_p 0.8", session)
    assert session.top_p == 0.8
    handle_slash_command("/seed 1234", session)
    assert session.seed == 1234

    # Add mock conversation turns
    session.messages.append(ChatMessage(role="user", content="Turn 1"))
    session.messages.append(ChatMessage(role="assistant", content="Answer 1"))
    session.messages.append(ChatMessage(role="user", content="Turn 2"))
    session.messages.append(ChatMessage(role="assistant", content="Answer 2"))

    # /undo removes last pair
    handle_slash_command("/undo", session)
    assert len(session.messages) == 3  # system + Turn 1 + Answer 1

    # /save
    transcript_file = tmp_path / "slash_saved.md"
    handle_slash_command(f"/save {transcript_file}", session)
    assert transcript_file.exists()

    # /clear resets turns
    handle_slash_command("/clear", session)
    assert len(session.messages) == 1  # Only system prompt remains

    # /load restores turns
    handle_slash_command(f"/load {transcript_file}", session)
    assert len(session.messages) >= 2

    # /toolbar toggle
    handle_slash_command("/toolbar off", session)
    assert session.toolbar_enabled is False
    handle_slash_command("/toolbar on", session)
    assert session.toolbar_enabled is True

    # /exit and /quit return True (signal exit)
    assert handle_slash_command("/exit", session) is True
    assert handle_slash_command("/quit", session) is True


# =====================================================================
# 5. RENDERER & INTERRUPT TESTS
# =====================================================================


def test_renderer_plain_and_interrupt() -> None:
    renderer = ChatRenderer(plain=True)

    def normal_stream():
        yield "Hello "
        yield "world!"

    text, interrupted = renderer.render_stream(normal_stream())
    assert text == "Hello world!"
    assert interrupted is False

    def interrupted_stream():
        yield "Part 1 "
        raise KeyboardInterrupt()

    text_intr, interrupted_flag = renderer.render_stream(interrupted_stream())
    assert text_intr == "Part 1 "
    assert interrupted_flag is True


# =====================================================================
# 6. REPL SESSION LIFECYCLE & CONTEXT PRUNING
# =====================================================================


def test_session_context_pressure_and_pruning(mock_server: str) -> None:
    client = ChatClient(base_url=mock_server)
    plan = make_dummy_plan()
    session = ChatSession(
        model_name="test-model",
        client=client,
        plan=plan,
        is_attached=True,
    )

    # Context limit is 4096 tokens (~16,384 chars)
    # Add messages exceeding 95% of 4096 tokens (over 15,500 chars)
    long_text = "A" * 6000
    session.messages.append(ChatMessage(role="user", content=long_text))
    session.messages.append(ChatMessage(role="assistant", content=long_text))
    session.messages.append(ChatMessage(role="user", content=long_text))
    session.messages.append(ChatMessage(role="assistant", content=long_text))

    initial_len = len(session.messages)
    session.check_context_window_pressure()
    # Oldest turn pair should have been automatically pruned
    assert len(session.messages) < initial_len
    assert session.messages[0].role == "system"


def test_session_end_to_end_with_mock_backend(mock_server: str, tmp_path: Path) -> None:
    adapter = MockBackendAdapter()
    plan = make_dummy_plan()
    proc = adapter.launch(plan, port=9999)

    client = ChatClient(base_url=mock_server)
    session = ChatSession(
        model_name="test-model",
        client=client,
        adapter=adapter,
        backend_proc=proc,
        plan=plan,
        auto_save_transcripts=False,
    )

    # Send prompt
    reply = session.send_prompt("Hello there")
    assert reply == "Hello from RAMLLM mock backend!"
    assert len(session.messages) == 3  # system, user, assistant

    # Replan attached
    handle_slash_command("/ctx 8192", session)

    # Shutdown cleanly
    session.shutdown()
    assert adapter.stopped is True
    assert_clean_process_state()


def test_chat_history_initialization() -> None:
    history = get_chat_history()
    assert history is not None
