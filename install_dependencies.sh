#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

WHL_DIR="./whl_packages"

# Create directory for wheel packages if it does not exist
if [ ! -d "${WHL_DIR}" ]; then
    echo "Creating directory ${WHL_DIR} for storing .whl files..."
    mkdir -p ${WHL_DIR}
else
    echo "Directory ${WHL_DIR} already exists. Skipping creation..."
fi

# Function to download a .whl file if it does not already exist
download_whl() {
    local url=$1
    local filepath=${WHL_DIR}/$(basename ${url})

    if [ ! -f "${filepath}" ]; then
        echo "Downloading ${url}..."
        wget -P ${WHL_DIR} ${url}
    else
        echo "File ${filepath} already exists. Skipping download..."
    fi
}

# ===============================
# Step 1: Install PyTorch
# ===============================
echo "Installing PyTorch 2.5.1 with CUDA 11.8..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# ===============================
# Step 2: Install general dependencies
# ===============================
if [ -f requirements.txt ]; then
    echo "Installing general dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping general dependency installation..."
fi

# ===============================
# Step 3: Install PyG-related wheel packages
# ===============================
echo "Downloading and installing PyG-related .whl dependencies into ${WHL_DIR}..."

download_whl https://data.pyg.org/whl/torch-2.5.0%2Bcu118/pyg_lib-0.4.0%2Bpt25cu118-cp312-cp312-linux_x86_64.whl
pip install ${WHL_DIR}/pyg_lib-0.4.0+pt25cu118-cp312-cp312-linux_x86_64.whl

download_whl https://data.pyg.org/whl/torch-2.5.0%2Bcu118/torch_cluster-1.6.3%2Bpt25cu118-cp312-cp312-linux_x86_64.whl
pip install ${WHL_DIR}/torch_cluster-1.6.3+pt25cu118-cp312-cp312-linux_x86_64.whl

download_whl https://data.pyg.org/whl/torch-2.5.0%2Bcu118/torch_scatter-2.1.2%2Bpt25cu118-cp312-cp312-linux_x86_64.whl
pip install ${WHL_DIR}/torch_scatter-2.1.2+pt25cu118-cp312-cp312-linux_x86_64.whl

download_whl https://data.pyg.org/whl/torch-2.5.0%2Bcu118/torch_sparse-0.6.18%2Bpt25cu118-cp312-cp312-linux_x86_64.whl
pip install ${WHL_DIR}/torch_sparse-0.6.18+pt25cu118-cp312-cp312-linux_x86_64.whl

download_whl https://data.pyg.org/whl/torch-2.5.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt25cu118-cp312-cp312-linux_x86_64.whl
pip install ${WHL_DIR}/torch_spline_conv-1.2.2+pt25cu118-cp312-cp312-linux_x86_64.whl

# ===============================
# Step 4: Install rxnfp and transformers
# ===============================
echo "Installing rxnfp (no dependencies) and transformers==4.43.4..."
pip install rxnfp --no-deps
pip install transformers==4.43.4

# ===============================
# Step 5: Install UniDock
# ===============================
echo "Installing UniDock (via conda + pip)..."

# Load conda shell functions to allow activate/deactivate inside script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Install UniDock via conda
echo "Installing unidock=1.1.2 from conda-forge (this step may take several minutes)..."
conda install -y -c conda-forge unidock=1.1.2

# Reactivate the environment (as recommended for UniDock setup)
echo "Reactivating conda environment 'spacegfn'..."
conda deactivate
conda activate spacegfn

# Install UniDock tools and related packages
echo "Installing openbabel-wheel and UniDock tools..."
pip install openbabel-wheel
pip install git+https://github.com/dptech-corp/Uni-Dock.git@1.1.2#subdirectory=unidock_tools

echo "✅ All dependencies have been successfully installed in conda environment 'spacegfn'"