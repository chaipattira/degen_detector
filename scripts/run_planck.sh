#!/bin/bash
#SBATCH --job-name=degen  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1            # Number of tasks
#SBATCH --time=2:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name

cd /home/x-ctirapongpra/scratch/
module load anaconda
conda activate degen_detector

python3 /home/x-ctirapongpra/scratch/degen_detector/scripts/run_planck_analysis.py --output-dir /home/x-ctirapongpra/scratch/degen_detector/outputs/planck --r2-threshold 1.0