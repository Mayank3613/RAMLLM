#!/usr/bin/env bash
# ==============================================================================
# RAMLLM Interactive Demonstration Script
# ==============================================================================
# This script demonstrates core RAMLLM workflows:
#   1. System & hardware diagnostic profiling (ramllm doctor)
#   2. Synthetic model fixture generation (<5 MB valid GGUF)
#   3. Adaptive memory-tier planning on local system
#   4. Cross-hardware simulation (CPU-only constrained laptop)
#   5. Verified dry-run execution
#   6. Programmatic Python SDK invocation
#
# Usage:
#   ./scripts/demo.sh [--non-interactive] [--delay <seconds>]
# ==============================================================================

set -euo pipefail

# Text styling
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
MAGENTA="\033[0;35m"
RESET="\033[0m"

NON_INTERACTIVE=false
DELAY=1.5

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --non-interactive|-y)
      NON_INTERACTIVE=true
      shift
      ;;
    --delay|-d)
      DELAY="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--non-interactive] [--delay <seconds>]"
      exit 0
      ;;
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done

pause() {
  if [ "$NON_INTERACTIVE" = false ]; then
    echo ""
    echo -e "${YELLOW}Press [Enter] to continue...${RESET}"
    read -r
  else
    sleep "$DELAY"
  fi
}

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Locate python and ramllm
if [ -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

echo -e "${BLUE}${BOLD}======================================================================${RESET}"
echo -e "${BLUE}${BOLD}                 RAMLLM Interactive Demonstration                     ${RESET}"
echo -e "${BLUE}${BOLD}       RAM-adaptive planning & enforcement layer for local LLMs       ${RESET}"
echo -e "${BLUE}${BOLD}======================================================================${RESET}"
echo ""

# Step 0: Ensure synthetic fixture model exists
FIXTURE_MODEL="${PROJECT_ROOT}/tests/fixtures/models/synthetic_tiny.gguf"
if [ ! -f "${FIXTURE_MODEL}" ]; then
  echo -e "${CYAN}▶ Preparing lightweight synthetic GGUF model fixture...${RESET}"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/create_fixture_gguf.py" --output "${FIXTURE_MODEL}"
  echo ""
fi

# Step 1: Diagnostics and System Probing
echo -e "${MAGENTA}${BOLD}[Step 1/5] Hardware Diagnostic & Enforcement Check${RESET}"
echo -e "Running ${BOLD}ramllm doctor${RESET} to probe CPU, RAM, VRAM, and OS enforcement level..."
pause
"${PYTHON}" -m ramllm doctor
echo ""

# Step 2: Planning on Local Hardware
echo -e "${MAGENTA}${BOLD}[Step 2/5] Adaptive Planning on Local Hardware${RESET}"
echo -e "Running ${BOLD}ramllm plan ${FIXTURE_MODEL} --details${RESET}..."
pause
"${PYTHON}" -m ramllm plan "${FIXTURE_MODEL}" --details
echo ""

# Step 3: Cross-Hardware Simulation
echo -e "${MAGENTA}${BOLD}[Step 3/5] Cross-Hardware Simulation (8GB CPU-Only Laptop)${RESET}"
echo -e "Simulating plan on a constrained 8 GB RAM laptop with no GPU:"
echo -e "${CYAN}Notice Flash Attention V-cache downgrade rule and CPU bandwidth modeling.${RESET}"
pause
"${PYTHON}" -m ramllm plan "${FIXTURE_MODEL}" \
  --hw-json "${PROJECT_ROOT}/tests/fixtures/hardware/laptop_8gb_no_gpu.json"
echo ""

# Step 4: Dry-Run Execution & Verification
echo -e "${MAGENTA}${BOLD}[Step 4/5] Verified Dry-Run Engine Launch${RESET}"
echo -e "Testing ${BOLD}ramllm run --dry-run${RESET} to verify engine arguments without spawning process:"
pause
"${PYTHON}" -m ramllm run "${FIXTURE_MODEL}" --dry-run
echo ""

# Step 5: Programmatic Python Library API
echo -e "${MAGENTA}${BOLD}[Step 5/5] Programmatic Python SDK Usage${RESET}"
echo -e "Executing inline Python snippet importing official public symbols:"
echo -e "${CYAN}from ramllm import probe_hardware, analyze_gguf, create_execution_plans, UserRequest${RESET}"
pause
"${PYTHON}" -c "
from ramllm import probe_hardware, analyze_gguf, create_execution_plans, UserRequest
hw = probe_hardware()
print(f'  ✔ Probed Hardware : {hw.ram_total_bytes / (1024**3):.1f} GB RAM, {len(hw.gpus)} GPU(s)')
model = analyze_gguf('${FIXTURE_MODEL}')
print(f'  ✔ Parsed GGUF     : {model.arch} ({model.n_layers} layers, {model.n_params:,} parameters)')
req = UserRequest(model=str(model.path))
plans = create_execution_plans(model, hw, req)
p = plans[0]
print(f'  ✔ Generated Plans : {len(plans)} candidate execution strategies evaluated')
print(f'  ✔ Top Plan Verdict: {p.verdict} (Bottleneck: {p.speed.bottleneck})')
print(f'  ✔ Estimated Decode: {p.speed.decode_tok_s_low:.0f}–{p.speed.decode_tok_s_high:.0f} tokens/second')
"
echo ""

echo -e "${GREEN}${BOLD}======================================================================${RESET}"
echo -e "${GREEN}${BOLD}✔ Demonstration completed successfully!${RESET}"
echo -e "Learn more at: ${CYAN}https://github.com/Mayank3613/CodeBrix${RESET}"
echo -e "${GREEN}${BOLD}======================================================================${RESET}"
