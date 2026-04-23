#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fi_swin_tiny
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/gpfs/mariana/home/svloor/Documents/ViT/run_swin_tiny_%j.out
#SBATCH --error=/gpfs/mariana/home/svloor/Documents/ViT/run_swin_tiny_%j.err

# Swin-tiny has 4 hierarchical stages (2+2+6+2 = 12 blocks total).
# Injecting across all blocks of each component type.
# Each run records which specific layer was hit so per-block analysis
# can be done in post-processing from the fault layer names in the JSON.
# 6000 repeats gives ~500 samples per block on average.

source /gpfs/mariana/home/svloor/Documents/ViT/.venv311/bin/activate
cd /gpfs/mariana/home/svloor/Documents/ViT/
export PYTHONPATH=/gpfs/mariana/home/svloor/Documents/ViT/src:$PYTHONPATH
mkdir -p new_runs

OUT=new_runs/swin_tiny_fi.json

# Save fault-free logits
python -m src.cli --model swin_tiny --max_batches 100 save --logits

# Fault-free baseline accuracy
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 100 --repeat 1 -o $OUT fi --faults 0

python -m src.cli --model swin_tiny --batch_size 100 --max_batches 100 --repeat 6000 -o $OUT fi --faults 1 --layers qkv  --component qkv
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 100 --repeat 6000 -o $OUT fi --faults 1 --layers proj --component proj
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 100 --repeat 6000 -o $OUT fi --faults 1 --layers fc1  --component fc1
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 100 --repeat 6000 -o $OUT fi --faults 1 --layers fc2  --component fc2
