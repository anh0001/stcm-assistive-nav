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

echo -e "${YELLOW}Using PYTHONUSERBASE=${PYTHONUSERBASE}${NC}"
echo -e "${YELLOW}Add 'export PYTHONUSERBASE=${PYTHONUSERBASE}' to your shell profile for future terminals.${NC}"

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

echo -e "\n${YELLOW}Step 3: Installing GroundingDINO${NC}"
if [ -d "$HOME/GroundingDINO" ]; then
    echo -e "${YELLOW}Existing GroundingDINO repo detected. Pulling latest...${NC}"
    git -C "$HOME/GroundingDINO" pull --ff-only
else
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$HOME/GroundingDINO"
fi
pip_user --no-build-isolation "$HOME/GroundingDINO"

echo -e "\n${YELLOW}Step 4: Installing MobileSAM${NC}"
pip_user git+https://github.com/ChaoningZhang/MobileSAM.git

echo -e "\n${YELLOW}Step 5: Installing STCM Python dependencies${NC}"
pip_user -r "$SCRIPT_DIR/stcm/requirements.txt"

echo -e "\n${YELLOW}Step 6: Installing ROS 2 dependencies via rosdep${NC}"
source /opt/ros/humble/setup.bash
rosdep install --from-paths "$SCRIPT_DIR/stcm" --ignore-src -y

echo -e "\n${YELLOW}Step 7: Building ROS 2 package${NC}"
cd "$SCRIPT_DIR"
colcon build --packages-select stcm

echo -e "\n${YELLOW}Step 8: Verifying imports${NC}"
"$PYTHON_BIN" - <<'VERIFY'
import torch
from groundingdino.util.inference import load_model
import mobile_sam
print("Torch:", torch.__version__)
print("GroundingDINO + MobileSAM import OK")
VERIFY

echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"
cat <<EOF

Next steps for each terminal:
1. export PYTHONUSERBASE="$PYTHONUSERBASE"    # add to ~/.bashrc for convenience
2. source /opt/ros/humble/setup.bash
3. source $(dirname "$0")/install/setup.bash

Optional helper alias:
    echo 'export PYTHONUSERBASE="$PYTHONUSERBASE"' >> ~/.bashrc
    echo 'alias stcm_setup="export PYTHONUSERBASE=$PYTHONUSERBASE && source /opt/ros/humble/setup.bash && source $SCRIPT_DIR/install/setup.bash"' >> ~/.bashrc

EOF
