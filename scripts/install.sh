#!/usr/bin/env sh
# RAMLLM installer script for macOS and Linux
# Usage: curl -LsSf https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.sh | sh

set -e

ASSUME_YES=0
TARGET_VERSION=""
UNINSTALL=0

print_help() {
    cat <<EOF
RAMLLM Installer

Usage:
  install.sh [options]

Options:
  -y, --yes          Assume yes to all prompts (non-interactive)
  -v, --version VER  Install a specific version of ramllm
  --uninstall        Uninstall ramllm tool
  -h, --help         Show this help message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes)
            ASSUME_YES=1
            shift
            ;;
        -v|--version)
            TARGET_VERSION="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            exit 1
            ;;
    esac
done

if [ "$UNINSTALL" -eq 1 ]; then
    echo "Uninstalling RAMLLM..."
    if command -v uv >/dev/null 2>&1; then
        uv tool uninstall ramllm || true
    elif command -v pipx >/dev/null 2>&1; then
        pipx uninstall ramllm || true
    else
        python3 -m pip uninstall -y ramllm || true
    fi
    echo "RAMLLM has been uninstalled."
    exit 0
fi

echo "=============================================="
echo "          RAMLLM Fast Installer               "
echo "=============================================="

# 1. Detect or provision uv
if ! command -v uv >/dev/null 2>&1; then
    echo "RAMLLM uses 'uv' to manage its standalone environment without requiring manual Python setup."
    INSTALL_UV="y"
    if [ "$ASSUME_YES" -eq 0 ]; then
        printf "Install uv now? [Y/n]: "
        read -r choice < /dev/tty || choice="y"
        case "$choice" in
            [nN]*) INSTALL_UV="n" ;;
            *) INSTALL_UV="y" ;;
        esac
    fi

    if [ "$INSTALL_UV" = "y" ]; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
fi

# 2. Install ramllm via tool manager
GIT_URL="git+https://github.com/Mayank3613/RAMLLM.git"
if [ -n "$TARGET_VERSION" ]; then
    VERSION_TAG="v${TARGET_VERSION#v}"
    GIT_SPEC="${GIT_URL}@${VERSION_TAG}"
    PYPI_SPEC="ramllm==$TARGET_VERSION"
else
    GIT_SPEC="$GIT_URL"
    PYPI_SPEC="ramllm"
fi

INSTALLED=0
if command -v uv >/dev/null 2>&1; then
    echo "Installing RAMLLM with uv tool..."
    if ! uv tool install --force "$PYPI_SPEC" 2>/dev/null; then
        echo "Note: Package '$PYPI_SPEC' not on PyPI registry; installing directly from GitHub repository..."
        uv tool install --force "$GIT_SPEC"
    fi
    INSTALLED=1
elif command -v pipx >/dev/null 2>&1; then
    echo "Installing RAMLLM with pipx..."
    if ! pipx install --force "$PYPI_SPEC" 2>/dev/null; then
        echo "Note: Package '$PYPI_SPEC' not on PyPI registry; installing directly from GitHub repository..."
        pipx install --force "$GIT_SPEC"
    fi
    INSTALLED=1
elif command -v python3 >/dev/null 2>&1; then
    echo "Warning: uv or pipx not found. Falling back to pip install --user..."
    if ! python3 -m pip install --user --upgrade "$PYPI_SPEC" 2>/dev/null; then
        echo "Note: Package '$PYPI_SPEC' not on PyPI registry; installing directly from GitHub repository..."
        python3 -m pip install --user --upgrade "$GIT_SPEC"
    fi
    INSTALLED=1
else
    echo "Error: Neither uv, pipx, nor python3 were found on your system." >&2
    echo "Please install uv (https://github.com/astral-sh/uv) or Python 3.11+ and retry." >&2
    exit 1
fi

# 3. PATH check
BIN_FOUND=0
if command -v ramllm >/dev/null 2>&1; then
    BIN_FOUND=1
fi

if [ "$BIN_FOUND" -eq 0 ]; then
    echo ""
    echo "Notice: 'ramllm' binary directory is not on your current shell PATH."
    echo "Add one of the following to your shell profile (~/.zshrc, ~/.bashrc):"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
    echo "Then run: source ~/.zshrc (or ~/.bashrc)"
fi

echo ""
echo "✔ RAMLLM installation complete!"
echo "Next step: Run 'ramllm setup'"
