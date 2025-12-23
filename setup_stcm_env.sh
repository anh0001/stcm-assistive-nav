#!/bin/bash
# STCM Environment Setup Script (system Python + isolated PYTHONUSERBASE)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}STCM Environment Setup${NC}"
echo -e "${GREEN}====================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Require python3 and ROS Humble
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: python3 not found. Install system Python 3.10 (default on Ubuntu 22.04).${NC}"
    exit 1
fi
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo -e "${RED}Error: ROS 2 Humble not found at /opt/ros/humble/setup.bash${NC}"
    echo "Install ROS 2 Humble before running this script."
    exit 1
fi

PYTHON_BIN="$(command -v python3)"
DEFAULT_USERBASE="$HOME/.local/stcm_sys_py310"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$DEFAULT_USERBASE}"
mkdir -p "$PYTHONUSERBASE"
export PATH="$PYTHONUSERBASE/bin:$PATH"

echo -e "${YELLOW}Using PYTHONUSERBASE=${PYTHONUSERBASE}${NC}"
echo -e "${YELLOW}Add 'export PYTHONUSERBASE=${PYTHONUSERBASE}' to your shell profile for future terminals.${NC}"
echo -e "${YELLOW}Add 'export PATH=${PYTHONUSERBASE}/bin:\\$PATH' so user-installed tools are on PATH.${NC}"

pip_user() {
    "$PYTHON_BIN" -m pip install --user "$@"
}

echo -e "\n${YELLOW}Step 1: Upgrading pip in the isolated user base${NC}"
"$PYTHON_BIN" -m pip install --upgrade --user pip

echo -e "\n${YELLOW}Step 2: Installing PyTorch 2.4.0 (CUDA 12.1 wheels)${NC}"
pip_user torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121

echo -e "\n${YELLOW}Verifying PyTorch installation...${NC}"
"$PYTHON_BIN" - <<'PYTORCH_CHECK'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
PYTORCH_CHECK

echo -e "\n${YELLOW}Step 3: Installing STCM Python dependencies (ros2-numpy requires numpy==1.24.2)${NC}"
pip_user -r "$SCRIPT_DIR/stcm/requirements.txt"
"$PYTHON_BIN" - <<'NUMPY_ROS2_CHECK'
import sys
import numpy
import importlib.metadata
from importlib.metadata import PackageNotFoundError

expected_numpy = "1.24.2"
expected_ros2_numpy = "0.0.5"

numpy_version = numpy.__version__
try:
    ros2_numpy_version = importlib.metadata.version("ros2-numpy")
except PackageNotFoundError:
    sys.exit("ros2-numpy is missing; pip likely bailed with a resolver error. Fix the install and rerun.")

print("NumPy pinned to:", numpy_version)
print("ros2-numpy version:", ros2_numpy_version)

if numpy_version != expected_numpy:
    sys.exit(f"Expected numpy {expected_numpy} but found {numpy_version}. Please rerun setup.")
if ros2_numpy_version != expected_ros2_numpy:
    sys.exit(f"Expected ros2-numpy {expected_ros2_numpy} but found {ros2_numpy_version}. Please rerun setup.")
NUMPY_ROS2_CHECK

echo -e "\n${YELLOW}Step 4: Installing STCM Planner core Python dependencies${NC}"
pip_user \
    spacy \
    numba \
    scipy \
    langchain==0.2.17 \
    langchain-core==0.2.43 \
    langchain-community==0.2.19 \
    langgraph==0.2.39 \
    typing_extensions
"$PYTHON_BIN" -m spacy download en_core_web_sm

echo -e "\n${YELLOW}Step 5: Installing STCM Planner provider dependencies${NC}"
pip_user \
    langchain-openai==0.1.20 \
    langchain-google-genai==1.0.10 \
    langchain-mistralai==0.1.13 \
    langchain-ollama==0.1.3 \
    google-generativeai==0.7.2 \
    mistralai==0.4.2

echo -e "\n${YELLOW}Step 6: Installing GroundingDINO${NC}"
if [ -d "$HOME/GroundingDINO" ]; then
    echo -e "${YELLOW}Existing GroundingDINO repo detected. Pulling latest...${NC}"
    git -C "$HOME/GroundingDINO" pull --ff-only
else
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$HOME/GroundingDINO"
fi
pip_user --no-build-isolation "$HOME/GroundingDINO"

echo -e "\n${YELLOW}Step 7: Installing MobileSAM${NC}"
pip_user git+https://github.com/ChaoningZhang/MobileSAM.git

echo -e "\n${YELLOW}Step 8: Installing ROS 2 dependencies via rosdep${NC}"
set +u  # ROS setup scripts reference unset vars such as AMENT_TRACE_SETUP_FILES
source /opt/ros/humble/setup.bash
set -u
rosdep install --from-paths "$SCRIPT_DIR/stcm" "$SCRIPT_DIR/stcm_planner" --ignore-src -y

echo -e "\n${YELLOW}Step 9: Cleaning previous colcon build artifacts (safe if absent)${NC}"
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/install" "$SCRIPT_DIR/log"

echo -e "\n${YELLOW}Step 10: Building ROS 2 packages${NC}"
cd "$SCRIPT_DIR"
colcon build --packages-select stcm stcm_planner

echo -e "\n${YELLOW}Step 11: Verifying imports${NC}"
"$PYTHON_BIN" - <<'VERIFY'
import torch
from groundingdino.util.inference import load_model
import mobile_sam
import spacy
import langchain
import langgraph
import mistralai
print("Torch:", torch.__version__)
spacy.load("en_core_web_sm")
print("GroundingDINO + MobileSAM import OK")
print("STCM Planner dependencies import OK")
VERIFY

echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"
cat <<EOF

Next steps for each terminal:
1. export PYTHONUSERBASE="$PYTHONUSERBASE"    # add to ~/.bashrc for convenience
2. export PATH="$PYTHONUSERBASE/bin:$PATH"
3. source /opt/ros/humble/setup.bash
4. source $(dirname "$0")/install/setup.bash

Optional helper alias:
    echo 'export PYTHONUSERBASE="$PYTHONUSERBASE"' >> ~/.bashrc
    echo 'export PATH="$PYTHONUSERBASE/bin:$PATH"' >> ~/.bashrc
    echo 'alias stcm_setup="export PYTHONUSERBASE=$PYTHONUSERBASE && export PATH=$PYTHONUSERBASE/bin:$PATH && source /opt/ros/humble/setup.bash && source $SCRIPT_DIR/install/setup.bash"' >> ~/.bashrc

EOF
