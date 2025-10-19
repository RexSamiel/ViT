#!/bin/bash
#SBATCH --partition=gpu             ### GPU partition
#SBATCH --job-name=ViT    ### Job name
#SBATCH --time=01:00:00            ### Max runtime HH:MM:SS
#SBATCH --nodes=1                   ### Single node
#SBATCH --ntasks=1                  ### Single task (for PyTorch)
#SBATCH --cpus-per-task=4           ### Number of CPU cores for dataloading
#SBATCH --gres=gpu:1                ### Request 1 GPU
#SBATCH --mem=20G                   ### RAM
source /gpfs/mariana/home/svloor/Documents/vit/common/vit_env/bin/activate
cd /gpfs/mariana/home/svloor/Documents/vit/
python main.py --faulty --metrics
