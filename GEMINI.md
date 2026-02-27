# SpaceGFN Project Overview

SpaceGFN is a generative drug discovery framework based on Generative Flow Networks (GFlowNets). it supports two primary operational modes: **Discovery** (de novo molecular construction) and **Editing** (optimization of existing molecules). The project utilizes Hydra for configuration management and UniDock for molecular docking and scoring.

## Core Technologies
- **Generative Framework:** GFlowNets (specifically Trajectory Balance loss).
- **Configuration:** [Hydra](https://hydra.cc/).
- **Cheminformatics:** [RDKit](https://www.rdkit.org/).
- **Deep Learning:** PyTorch, PyTorch Geometric (PyG), PyTorch Lightning.
- **Docking/Scoring:** UniDock, QSAR models (EGFR, FGFR1, SRC, VEGFR2).
- **Logging:** Weights & Biases (WandB), TensorBoard.

## Project Structure
- `config/`: Hydra configuration files for environment, policies, proxies, and logging.
- `gflownet/`: Core implementation of the GFlowNet agent, environments (`envs/`), policies (`policy/`), and utilities (`utils/`).
- `dataprocess/`: Data preparation scripts and Jupyter notebooks for Discovery and Editing modes.
- `data/`: Initial ligands, proteins, and templates for docking and generation.
- `whl_packages/`: Pre-compiled wheel packages for PyTorch Geometric dependencies.

## Setup & Installation
1. **Create Environment:**
   ```bash
   conda create --name spacegfn python==3.12.9
   conda activate spacegfn
   ```
2. **Install Dependencies:**
   ```bash
   bash install_dependencies.sh
   ```
   *Note: This script installs specific versions of PyTorch, PyG, and UniDock.*

## Execution & Training
The main entry point is `main.py`.

### Basic Execution
```bash
python main.py user.logdir.root=<path/to/logs>
```

### Modes of Operation
- **Discovery Mode:** Set `mode: "discovery"` in config or CLI. Used for constructing chemical spaces from building blocks.
- **Editing Mode:** Set `mode: "editing"` in config or CLI. Used for optimizing existing lead compounds using reaction templates (Edit Rule V1).

### Training Script
A shortcut for training is provided:
```bash
bash train.sh
```

## Development Conventions
- **Configuration:** Always use Hydra. Default settings are in `config/main.yaml`. User-specific overrides should be placed in `config/user/<username>.yaml`.
- **Environment:** The `BuildingBlock` environment (`gflownet/envs/building_block.py`) defines the state space and transitions.
- **Proxy:** Reward functions are defined as proxies in `gflownet/proxy/`.
- **Linting/Style:** Follow standard Python (PEP 8) practices. RDKit logging is typically disabled in `main.py`.

## Key Files
- `main.py`: Entry point for training and sampling.
- `gflownet/gflownet.py`: Implementation of the `GFlowNetAgent`.
- `config/main.yaml`: Central configuration for the entire framework.
- `install_dependencies.sh`: Critical for setting up the complex CUDA-enabled environment.
