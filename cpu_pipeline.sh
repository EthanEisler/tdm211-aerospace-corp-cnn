#!/bin/bash
#SBATCH --job-name=cnn_cpu_pipeline
#SBATCH --account=tdm-2601
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=cpu_pipeline_%j.out
#SBATCH --error=cpu_pipeline_%j.err

set -euo pipefail

cd /anvil/projects/tdm-2601/cnn
source .venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export DSET_FIOT_HIGHWAY2=/anvil/projects/tdm-2601/fiot_highway2_f32

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
python --version

echo "=== Running time.py ==="
python -u time.py

echo "=== Training CNN for 5 epochs ==="
python -u train.py --epochs 5

echo "=== Running test.py ==="
python -u test.py

echo "All programs completed: $(date)"

