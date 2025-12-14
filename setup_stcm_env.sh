#!/bin/bash
# STCM Environment Setup Script
# This script automates the installation of STCM with proper conda environment setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}STCM Environment Setup${NC}"
echo -e "${GREEN}====================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if ROS 2 Humble is available
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo -e "${RED}Error: ROS 2 Humble not found${NC}"
    echo "Please install ROS 2 Humble first:"
    echo "https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

# Create conda environment
echo -e "\n${YELLOW}Step 1: Creating conda environment 'stcm_env'${NC}"
if conda env list | grep -q "^stcm_env "; then
    echo -e "${YELLOW}Environment 'stcm_env' already exists. Skipping creation.${NC}"
else
    conda create -n stcm_env python=3.10 -y
    echo -e "${GREEN}Environment created successfully${NC}"
fi

# Activate the environment
echo -e "\n${YELLOW}Step 2: Activating conda environment${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stcm_env

# Ensure pip is properly installed in the conda environment
echo -e "\n${YELLOW}Step 2.5: Ensuring pip is properly installed in conda environment${NC}"
conda install pip -y

# Install PyTorch with CUDA support
echo -e "\n${YELLOW}Step 3: Installing PyTorch 2.4.0 with CUDA 12.1${NC}"
pip install \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
echo -e "\n${YELLOW}Verifying PyTorch installation...${NC}"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# Install GroundingDINO
echo -e "\n${YELLOW}Step 4: Installing GroundingDINO${NC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$HOME/GroundingDINO" ]; then
    echo -e "${YELLOW}GroundingDINO already cloned. Using existing installation.${NC}"
    cd "$HOME/GroundingDINO"
    git pull
else
    cd "$HOME"
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
fi
# Install without build isolation to use already installed torch
pip install --no-build-isolation -e .

# Install MobileSAM from GitHub
echo -e "\n${YELLOW}Step 5: Installing MobileSAM${NC}"
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Install remaining dependencies
echo -e "\n${YELLOW}Step 6: Installing remaining Python dependencies${NC}"
cd "$SCRIPT_DIR/stcm"

# Install core dependencies first to avoid version conflicts
echo "Installing core numerical and build dependencies..."
pip install numpy==1.26.4 setuptools wheel Cython

# Install specific versioned packages to avoid backtracking
echo "Installing pinned versions..."
pip install \
  supervision==0.18.0 \
  transformers==4.44.0 \
  huggingface-hub==0.25.0

# Install remaining packages
echo "Installing remaining dependencies..."
pip install \
  networkx \
  tqdm \
  opencv-python \
  open3d \
  scikit-image \
  shapely \
  Pillow \
  matplotlib \
  ftfy \
  regex \
  requests \
  PyYAML \
  packaging \
  chardet \
  absl-py \
  ros2-numpy

# Install ROS dependencies
echo -e "\n${YELLOW}Step 7: Installing ROS 2 dependencies${NC}"
source /opt/ros/humble/setup.bash
cd "$SCRIPT_DIR"
rosdep install --from-paths stcm --ignore-src -y

# Build the ROS 2 package
echo -e "\n${YELLOW}Step 8: Building ROS 2 package${NC}"
colcon build --packages-select stcm

# Verify installation
echo -e "\n${YELLOW}Step 9: Verifying installation${NC}"
python -c "import torch; from groundingdino.util.inference import load_model; print('GroundingDINO import OK')"

# Success message
echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"
echo -e "\nTo use STCM in a new terminal, run:"
echo -e "${YELLOW}conda activate stcm_env${NC}"
echo -e "${YELLOW}source /opt/ros/humble/setup.bash${NC}"
echo -e "${YELLOW}source $(dirname "$0")/install/setup.bash${NC}"
echo -e "\nOr add this alias to your ~/.bashrc:"
echo -e "${YELLOW}alias stcm_setup='conda activate stcm_env && source /opt/ros/humble/setup.bash && source $(dirname "$0")/install/setup.bash'${NC}"
