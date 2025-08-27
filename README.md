# 🌍 SIFLUXCOM Synthetic Project

This repository contains code, Jupyter notebooks, and environment setup for running experiments on meteorological and Plant Functional Type (PFT) data.  

It includes workflows for:
- ✅ Dataset preparation  
- ✅ Pre-training and fine-tuning models  
- ✅ Forward simulations with meteorological & PFT data  
- ✅ HPC job submission scripts  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/XinYuScience/SIFLUXCOM_synthetic.git
cd SIFLUXCOM_synthetic
```

### 2. Create the Conda Environment
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.  

Create the environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

### 3. Activate the Environment
```bash
conda activate sifluxcom
```

*(Replace `sifluxcom` with the actual environment name defined in `environment.yml` under `name:` if different.)*

### 4. Verify Installation
```bash
python --version
conda list
```

---

## 📂 Repository Structure
```
SIFLUXCOM_synthetic/
├── data/                     # Input datasets (not tracked by git)
├── outputs/                  # Experiment results and logs (not tracked by git)
├── dataset_prep.ipynb        # Notebook: dataset preparation
├── default_forward_*.ipynb   # Default forward simulations
├── default_training_*.ipynb  # Default training runs
├── fine_tuning_*.ipynb       # Fine-tuning experiments
├── pre_training_*.ipynb      # Pre-training experiments
├── model.py                  # Model definitions
├── utils.py                  # Utility functions
├── submit_job.sh             # HPC SLURM job submission script
├── environment.yml           # Conda environment specification
└── README.md                 # Project documentation
```

---

## ⚠️ Notes
- Large data files (`data/`, `outputs/`) are not stored in this repository.  
- Please prepare your own `data/` directory or request access if applicable.  
- Some notebooks assume you are running on an HPC with **SLURM** job scheduling.

---

## ✨ Citation
If you use this repository in your research, please cite appropriately (add your reference here).  

---

👩‍💻 Maintainer: **XinYuScience**  
📬 Contributions, issues, and suggestions are welcome!
