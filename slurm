#!/bin/bash
#SBATCH -A YOUR_PROJECT_ACCOUNT         # ← Replace with your actual account
#SBATCH -J tomo_recon                   # Job name
#SBATCH --nodes=1                       # 1 node
#SBATCH --gres=gpu:1                    # ← ONLY 1 GPU (not 8!)
#SBATCH --ntasks-per-node=1             # ← ONLY 1 task
#SBATCH --cpus-per-task=8              # More CPUs for your single task
#SBATCH -t 00:30:00                     # Time limit
#SBATCH -q debug                        # Queue
#SBATCH -o tomo-%j.out                  # Output file
#SBATCH -e tomo-%j.err                  # Error file

# ============================================================
# 1. LOAD MODULES (simplified for single GPU)
# ============================================================
module load PrgEnv-gnu
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

module unload darshan-runtime

# ============================================================
# 2. SETUP PYTHON ENVIRONMENT
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/world-shared/topcicekd/pytorch_env

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# ============================================================
# 3. SINGLE GPU OPTIMIZATION FLAGS
# ============================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=hybrid

# Important: Reduce memory fragmentation for single GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================
# 4. RUN YOUR SINGLE-GPU CODE
# ============================================================

echo "=========================================="
echo "Starting SINGLE GPU tomography reconstruction"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Run your single-GPU code
time python main_simple.py

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

# Show output files
echo "Output files created in frontier_output/:"
ls -lh frontier_output/ 2>/dev/null || echo "No output directory found"
