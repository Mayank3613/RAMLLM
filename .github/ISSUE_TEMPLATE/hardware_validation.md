---
name: Hardware Benchmark Validation
about: Submit real-world benchmark data to improve calibration accuracy
title: "[VALIDATION] "
labels: ["validation", "benchmark"]
assignees: []
---

## Hardware Configuration
- **Hardware Profile / Model**: (e.g. MacBook Pro M3 Pro, Custom Desktop RTX 3080)
- **CPU**: (e.g., AMD Ryzen 9 5900X, Apple M3)
- **RAM Total & Measured Bandwidth**: (e.g., 32 GB DDR4 @ 38.2 GB/s)
- **GPU Model & VRAM**: (e.g., NVIDIA RTX 3080 10GB @ 760 GB/s)
- **Operating System**: (e.g., Linux 6.8, macOS 14.5, Windows 11)

## Benchmark Results (`ramllm bench`)
Please run:
```bash
ramllm bench <model.gguf> --export validation_report.json
```

Paste the summary table or attach `validation_report.json`:

```text
Paste terminal table output from `ramllm bench` here
```

## Prediction Accuracy
- **Model Architecture & Quant**: (e.g., Qwen2.5-7B Q4_K_M)
- **Predicted tok/s Range**: [Low, High]
- **Measured tok/s**: Measured
- **Accuracy Ratio**: Measured / Midpoint

## Additional Notes
Any relevant environmental notes (e.g., background applications running, thermal throttling observed).
