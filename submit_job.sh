#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_out_err/job.out.%j
#SBATCH -e ./job_out_err/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_gpu
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xyu@bgc-jena.mpg.de
#SBATCH --time=12:00:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
module load intel/21.2.0 impi/2021.2 cuda/11.2 python-waterboa/2024.06

# The following command replaces `conda init` for the current session
# without touching the .bashrc file:
eval "$(conda shell.bash hook)"

conda activate SIFLUXCOM_Xin

start_time=$(date +%s)

srun python3 ./default_training_meteo_ex.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $(($elapsed_time / 3600))h $((($elapsed_time / 60) % 60))m $(($elapsed_time % 60))s"
exit