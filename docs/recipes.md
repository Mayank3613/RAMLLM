# RAMLLM Integration Recipes

This guide covers real-world recipes for integrating RAMLLM into external applications, developer tools, and automated pipelines.

---

## 1. Connecting to Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is a popular web interface for local LLMs. You can easily connect it to a detached RAMLLM background server.

### Step 1: Start RAMLLM in background
```bash
ramllm serve qwen2.5-7b --detach --port 8080
```

### Step 2: Run Open WebUI in Docker
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8080/v1" \
  -e OPENAI_API_KEY="ramllm-local" \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Open `http://localhost:3000` in your web browser. Your active model will appear automatically in the model selector.

---

## 2. Using with the Official OpenAI Python SDK

RAMLLM serves standard OpenAI-compatible `/v1/chat/completions` endpoints.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="ramllm-local",  # Any string is accepted
)

response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {"role": "system", "content": "You are a concise programming assistant."},
        {"role": "user", "content": "How do I reverse a list in Python?"},
    ],
    temperature=0.7,
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```

---

## 3. Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="ramllm-local",
    model="llama-3.2-3b",
    temperature=0.2,
)

result = llm.invoke("Summarize the importance of memory bandwidth in 3 bullets.")
print(result.content)
```

---

## 4. Low-RAM Setup: Running on 8 GB Machines (e.g. 8 GB M1/M2/M3 MacBook or Thin Laptop)

When running on computers with 8 GB of total memory, ordinary tools like Ollama or LM Studio often freeze the OS or crash when browser tabs are open.

Follow these tips to get great performance on 8 GB machines:

```bash
# 1. Set RAMLLM memory usage to conservative (50-60% headroom reserved)
ramllm config set memory_usage conservative

# 2. Use 'fast' mode which quantizes the KV cache to q8_0/q4_0
ramllm chat llama-3.2-3b --mode fast

# 3. Request a modest context window (2048 or 4096 tokens)
ramllm chat qwen2.5-3b --context 2048
```

Models that run exceptionally well on 8 GB:
- `llama-3.2-3b` (Q4_K_M ~2.0 GB) — Fast, responsive, fits 100% in VRAM.
- `qwen2.5-3b` (Q4_K_M ~2.0 GB) — Superb coding and reasoning in compact footprint.
- `smollm2-1.7b` (Q4_K_M ~1.0 GB) — Ultra-lightweight assistant for quick tasks.

---

## 5. Multi-GPU Setups (e.g. Dual 8 GB RTX 3070 / 4060)

If your machine has multiple GPUs, RAMLLM automatically splits layer tensors across both devices to pool your VRAM:

```bash
# Specify comma-separated GPU indices and budgets
ramllm plan deepseek-r1-distill-qwen-14b --gpus 0,1 --vram 8GB,8GB

# Run chat with multi-GPU placement
ramllm chat deepseek-r1-distill-qwen-14b --gpus 0,1
```

RAMLLM calculates optimal split ratios based on the compute bandwidth and available free memory of each GPU.

---

## 6. Running a Persistent System Service with Systemd (Linux)

To run RAMLLM as an unattended background service on boot:

Create `/etc/systemd/system/ramllm.service`:

```ini
[Unit]
Description=RAMLLM Model Inference Server
After=network.target

[Service]
Type=simple
User=yourusername
ExecStart=/usr/local/bin/ramllm serve qwen2.5-7b --port 8080 --host 127.0.0.1
Restart=always
RestartSec=5
Environment=RAMLLM_HOME=/home/yourusername/.ramllm

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ramllm
```

Check status and logs:
```bash
sudo systemctl status ramllm
ramllm ps
ramllm dashboard
```
