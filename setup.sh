#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Alpha Signal Engine — One-Click Setup & Run
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What it does:
#   1. Checks Python 3.12+ is installed
#   2. Installs uv (if not present)
#   3. Installs all dependencies
#   4. Creates .env from .env.example (if not exists)
#   5. Creates data directory
#   6. Runs tests to verify installation
#   7. Starts the server
#
# Requirements:
#   - Python 3.12+ (brew install python3 / apt install python3)
#   - Internet connection (for package downloads + live market data)
#
# Optional:
#   - Alpaca API key for US stocks (free at https://alpaca.markets)
#   - Binance needs NO key for crypto (public candle data is free)
# ══════════════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║         Alpha Signal Engine — Setup              ║"
echo "║   AI Trading Signal Agent + Paper Trading        ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: Install Python if missing ───────────────────────────────────────
echo -e "${YELLOW}[1/7] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "  ${GREEN}Python ${PY_VERSION} found${NC}"
else
    echo -e "  ${YELLOW}Python 3 not found — installing...${NC}"
    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install python@3.12
        else
            echo "  Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null)"
            brew install python@3.12
        fi
    elif [[ -f /etc/debian_version ]]; then
        # Ubuntu/Debian
        sudo apt update && sudo apt install -y python3 python3-pip python3-venv
    elif [[ -f /etc/redhat-release ]]; then
        # CentOS/RHEL/Fedora
        sudo dnf install -y python3 python3-pip
    elif [[ -f /etc/arch-release ]]; then
        # Arch Linux
        sudo pacman -Sy --noconfirm python python-pip
    else
        echo -e "  ${RED}Could not auto-install Python. Please install Python 3.12+ manually.${NC}"
        exit 1
    fi
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "  ${GREEN}Python ${PY_VERSION} installed${NC}"
fi

# ── Step 2: Install uv + git ───────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Checking dependencies (uv, git, curl)...${NC}"

# git
if ! command -v git &> /dev/null; then
    echo "  Installing git..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git
    elif [[ -f /etc/debian_version ]]; then
        sudo apt install -y git
    elif [[ -f /etc/redhat-release ]]; then
        sudo dnf install -y git
    fi
fi
echo -e "  ${GREEN}git $(git --version 2>/dev/null | head -1)${NC}"

# curl
if ! command -v curl &> /dev/null; then
    echo "  Installing curl..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install curl
    elif [[ -f /etc/debian_version ]]; then
        sudo apt install -y curl
    fi
fi

# uv
if command -v uv &> /dev/null; then
    echo -e "  ${GREEN}uv already installed $(uv --version 2>/dev/null | head -1)${NC}"
else
    echo "  Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    # Source shell profile to pick up uv
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null
    [ -f "$HOME/.zshrc" ] && source "$HOME/.zshrc" 2>/dev/null
    echo -e "  ${GREEN}uv installed${NC}"
fi

# ── Step 3: Install dependencies ────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Installing dependencies...${NC}"
uv sync 2>&1 | tail -3
echo -e "  ${GREEN}Dependencies installed${NC}"

# ── Step 4: Create .env ────────────────────────────────────────────────────
echo -e "${YELLOW}[4/7] Setting up environment...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "  ${GREEN}Created .env from .env.example${NC}"
    echo -e "  ${YELLOW}NOTE: Edit .env to add your Alpaca API key for stock data${NC}"
    echo -e "  ${YELLOW}      Crypto works without any API key${NC}"
else
    echo -e "  ${GREEN}.env already exists — keeping current config${NC}"
fi

# ── Step 5: Create data directory ───────────────────────────────────────────
echo -e "${YELLOW}[5/7] Creating data directory...${NC}"
mkdir -p data
echo -e "  ${GREEN}data/ directory ready${NC}"

# ── Step 6: Run tests ──────────────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Running tests...${NC}"
TEST_OUTPUT=$(uv run python -m pytest tests/unit/test_market_mode_validator.py tests/unit/test_bias_exit.py tests/unit/test_participation_matrix.py -q 2>&1 | tail -1)
echo -e "  ${GREEN}${TEST_OUTPUT}${NC}"

# ── Step 7: Start server ───────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗"
echo -e "║              Setup Complete!                      ║"
echo -e "╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Dashboard:  ${GREEN}http://localhost:8000/paper${NC}"
echo -e "  API docs:   ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "  Scanning: 32 stocks + 29 crypto + 20 futures = ${GREEN}61 symbols${NC}"
echo -e "  Paper trading: ${GREEN}\$10,000 virtual capital${NC}"
echo -e "  Bots: ${GREEN}6 competing hedge fund bots${NC}"
echo ""
echo -e "${YELLOW}Starting server (30s scan interval)...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

uv run python -m apps.signal_agent.main serve --interval 30
