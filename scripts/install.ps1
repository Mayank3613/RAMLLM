<#
.SYNOPSIS
    RAMLLM installer script for Windows PowerShell.
    Usage: irm https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.ps1 | iex
#>
param (
    [switch]$Yes = $false,
    [string]$Version = "",
    [switch]$Uninstall = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host @"
RAMLLM Windows Installer

Usage:
  install.ps1 [-Yes] [-Version <ver>] [-Uninstall] [-Help]

Options:
  -Yes          Assume yes to prompts
  -Version VER  Install specific version
  -Uninstall    Uninstall RAMLLM
  -Help         Show this help
"@
    exit 0
}

if ($Uninstall) {
    Write-Host "Uninstalling RAMLLM..."
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv tool uninstall ramllm
    } elseif (Get-Command pipx -ErrorAction SilentlyContinue) {
        pipx uninstall ramllm
    } else {
        python -m pip uninstall -y ramllm
    }
    Write-Host "RAMLLM has been uninstalled."
    exit 0
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "          RAMLLM Fast Windows Installer       " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Detect or provision uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "RAMLLM uses 'uv' for standalone environment management."
    $InstallUv = $true
    if (-not $Yes) {
        $choice = Read-Host "Install uv now? [Y/n]"
        if ($choice -match "^[nN]") {
            $InstallUv = $false
        }
    }

    if ($InstallUv) {
        Write-Host "Installing uv via official installer..."
        irm https://astral.sh/uv/install.ps1 | iex
        $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
    }
}

$pkgSpec = "ramllm"
if ($Version -ne "") {
    $pkgSpec = "ramllm==$Version"
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Installing RAMLLM with uv tool..."
    uv tool install --force $pkgSpec
} elseif (Get-Command pipx -ErrorAction SilentlyContinue) {
    Write-Host "Installing RAMLLM with pipx..."
    pipx install --force $pkgSpec
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "uv/pipx not found. Falling back to pip install --user..."
    python -m pip install --user --upgrade $pkgSpec
} else {
    Write-Error "Neither uv, pipx, nor python were found. Please install uv (https://astral.sh/uv) and retry."
    exit 1
}

if (-not (Get-Command ramllm -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Notice: 'ramllm' is not yet on your current PATH." -ForegroundColor Yellow
    Write-Host "Add the following to your User PATH or restart your terminal:"
    Write-Host "  $env:USERPROFILE\.local\bin"
}

Write-Host ""
Write-Host "RAMLLM installation complete!" -ForegroundColor Green
Write-Host "Next step: Run 'ramllm setup'" -ForegroundColor Cyan
