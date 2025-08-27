# 🌍 SIFLUXCOM Synthetic Project

-Workflow of the project
1) create synthetic dataset for different extrapolation cases based on the dataset era5_ssrd_vpd_2001-2020.zarr
2) default: train neural networks directly for the target
3) transfer learning: pre training on the proxy; fine tuning on the target
4) evaluate the results
5) bonus: i.think of other approaches ii. think of PFT replacements

Notes:
You could work with .ipynb to interactively run the script or submit jobs using submit_job.sh

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
├── data.tar.gz               # Input datasets
├── dataset_prep.ipynb        # Notebook: dataset preparation
├── default_forward_*.ipynb   # Default forward simulations
├── default_training_*.py     # example .py file
├── default_training_*.ipynb  # Default training runs
├── fine_tuning_*.ipynb       # Fine-tuning experiments
├── pre_training_*.ipynb      # Pre-training experiments
├── model.py                  # Model definitions
├── utils.py                  # Utility functions
├── submit_job.sh             # HPC SLURM job submission script
├── environment.yml           # Conda environment specification
└── README.md                 # Project documentation
```
