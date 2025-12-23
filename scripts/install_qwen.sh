#!/bin/bash
# Qwen Installation Script for STCM Assistive Navigation
# This script installs Ollama and pulls Qwen models for local LLM inference

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Qwen Installation for STCM${NC}"
echo -e "${GREEN}====================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check if Ollama is installed
echo -e "\n${YELLOW}Step 1: Checking for Ollama installation${NC}"
if command_exists ollama; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Ollama is already installed: ${OLLAMA_VERSION}${NC}"
else
    echo -e "${YELLOW}Ollama not found. Installing Ollama...${NC}"

    # Check if running with sudo is possible
    if command_exists sudo; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo -e "${RED}Error: sudo not available. Please install Ollama manually:${NC}"
        echo -e "${BLUE}Visit: https://github.com/ollama/ollama${NC}"
        exit 1
    fi

    # Verify installation
    if command_exists ollama; then
        echo -e "${GREEN}✓ Ollama installed successfully${NC}"
    else
        echo -e "${RED}Error: Ollama installation failed${NC}"
        exit 1
    fi
fi

# Step 2: Check if Ollama service is running
echo -e "\n${YELLOW}Step 2: Checking Ollama service${NC}"
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✓ Ollama service is running${NC}"
else
    echo -e "${YELLOW}Ollama service not running. Starting it...${NC}"

    # Try to start Ollama in the background
    if command_exists systemctl; then
        # If systemd is available, try to start the service
        sudo systemctl start ollama 2>/dev/null || true
    fi

    # Check if it's running now
    sleep 2
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}✓ Ollama service started${NC}"
    else
        echo -e "${YELLOW}Starting Ollama serve in background...${NC}"
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3

        if pgrep -x "ollama" > /dev/null; then
            echo -e "${GREEN}✓ Ollama serve started (PID: $(pgrep -x ollama))${NC}"
        else
            echo -e "${RED}Warning: Could not start Ollama service automatically${NC}"
            echo -e "${YELLOW}Please run 'ollama serve' in a separate terminal${NC}"
        fi
    fi
fi

# Step 3: Pull Qwen models
echo -e "\n${YELLOW}Step 3: Pulling Qwen models${NC}"
echo -e "${BLUE}Available models:${NC}"
echo -e "  1. deepseek-r1:7b (Qwen-based reasoning model) - ${YELLOW}Already configured${NC}"
echo -e "  2. qwen2.5:7b (7B parameters, ~4.7GB)"
echo -e "  3. qwen2.5:14b (14B parameters, ~9GB)"
echo -e "  4. qwen2.5:32b (32B parameters, ~20GB)"
echo -e "  5. qwen2.5:72b (72B parameters, ~43GB)"
echo ""

# Default model that's already configured in the codebase
DEFAULT_MODEL="deepseek-r1:7b"

# Declare associative array for model mapping
declare -A MODEL_MAP
MODEL_MAP[1]="deepseek-r1:7b"
MODEL_MAP[2]="qwen2.5:7b"
MODEL_MAP[3]="qwen2.5:14b"
MODEL_MAP[4]="qwen2.5:32b"
MODEL_MAP[5]="qwen2.5:72b"

# Ask user which models to install
echo -e "${YELLOW}Which model would you like to install?${NC}"
echo -e "Enter numbers (1-5) or model names, separated by spaces (or press Enter for default: 1):"
read -r MODELS_INPUT

# Use default if empty
if [ -z "$MODELS_INPUT" ]; then
    MODELS_TO_INSTALL=($DEFAULT_MODEL)
else
    # Parse input and convert numbers to model names
    MODELS_TO_INSTALL=()
    for INPUT in $MODELS_INPUT; do
        # Check if input is a number between 1-5
        if [[ "$INPUT" =~ ^[1-5]$ ]]; then
            MODELS_TO_INSTALL+=("${MODEL_MAP[$INPUT]}")
        else
            # Assume it's a model name
            MODELS_TO_INSTALL+=("$INPUT")
        fi
    done
fi

# Pull each model
for MODEL in "${MODELS_TO_INSTALL[@]}"; do
    echo -e "\n${YELLOW}Pulling ${MODEL}...${NC}"

    # Check if model already exists
    if ollama list | grep -q "$MODEL"; then
        echo -e "${GREEN}✓ Model ${MODEL} is already installed${NC}"

        # Ask if user wants to update
        echo -e "${YELLOW}Update ${MODEL} to latest version? (y/N):${NC}"
        read -r UPDATE_CHOICE
        if [[ "$UPDATE_CHOICE" =~ ^[Yy]$ ]]; then
            ollama pull "$MODEL"
            echo -e "${GREEN}✓ Model ${MODEL} updated${NC}"
        fi
    else
        # Pull the model
        if ollama pull "$MODEL"; then
            echo -e "${GREEN}✓ Model ${MODEL} installed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to install ${MODEL}${NC}"
        fi
    fi
done

# Step 4: Verify installation
echo -e "\n${YELLOW}Step 4: Verifying installation${NC}"
echo -e "${BLUE}Installed Qwen models:${NC}"
ollama list | grep -E "(qwen|deepseek-r1)" || echo "No Qwen models found"

# Step 5: Test the model
echo -e "\n${YELLOW}Step 5: Testing model${NC}"
TEST_MODEL="${MODELS_TO_INSTALL[0]}"
echo -e "${BLUE}Testing ${TEST_MODEL} with a simple query...${NC}"

TEST_OUTPUT=$(ollama run "$TEST_MODEL" "Say 'Hello from Qwen!' in one sentence." 2>&1 || echo "Test failed")
if [[ "$TEST_OUTPUT" == *"Hello"* ]] || [[ "$TEST_OUTPUT" == *"Qwen"* ]]; then
    echo -e "${GREEN}✓ Model test successful${NC}"
    echo -e "${BLUE}Response: ${TEST_OUTPUT}${NC}"
else
    echo -e "${YELLOW}Warning: Model test produced unexpected output${NC}"
    echo -e "${BLUE}Output: ${TEST_OUTPUT}${NC}"
fi

# Step 6: Display usage instructions
echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"

cat <<EOF

${BLUE}Usage Instructions:${NC}

1. ${YELLOW}Ensure Ollama is running:${NC}
   ${BLUE}ollama serve${NC}   # Run in separate terminal, or it auto-starts

2. ${YELLOW}List installed models:${NC}
   ${BLUE}ollama list${NC}

3. ${YELLOW}Test a model interactively:${NC}
   ${BLUE}ollama run ${TEST_MODEL}${NC}

4. ${YELLOW}Use in Python code:${NC}
   ${BLUE}from stcm_planner.llm_backend.llm_query_langchain import LLMQueryHandler
   from stcm_planner.llm_backend.enums import LanguageModel

   # For DeepSeek-R1 (already configured)
   handler = LLMQueryHandler(model=LanguageModel.R1_QWEN2)${NC}

5. ${YELLOW}Add more Qwen variants (optional):${NC}
   Edit: ${BLUE}stcm_planner/stcm_planner/llm_backend/enums.py${NC}

   Add to LanguageModel enum:
   ${BLUE}QWEN2_7B = "qwen2.5:7b"
   QWEN2_14B = "qwen2.5:14b"${NC}

   These will automatically work with the existing Ollama integration.

6. ${YELLOW}Environment setup for STCM:${NC}
   ${BLUE}export PYTHONUSERBASE="\$HOME/.local/stcm_sys_py310"
   source /opt/ros/humble/setup.bash
   source ./install/setup.bash${NC}

${YELLOW}Installed models:${NC}
$(ollama list | grep -E "(qwen|deepseek-r1)" || echo "  (none)")

${YELLOW}Useful Ollama commands:${NC}
  ${BLUE}ollama list${NC}              - List installed models
  ${BLUE}ollama pull <model>${NC}      - Download a model
  ${BLUE}ollama rm <model>${NC}        - Remove a model
  ${BLUE}ollama run <model>${NC}       - Run model interactively
  ${BLUE}ollama serve${NC}             - Start Ollama server

${YELLOW}Model sizes (approximate):${NC}
  deepseek-r1:7b  - ~4.7GB
  qwen2.5:7b      - ~4.7GB
  qwen2.5:14b     - ~9GB
  qwen2.5:32b     - ~20GB
  qwen2.5:72b     - ~43GB

EOF
