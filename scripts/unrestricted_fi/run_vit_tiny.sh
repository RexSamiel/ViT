#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fi_vit_tiny
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/gpfs/mariana/home/svloor/Documents/ViT/run_vit_tiny_%j.out
#SBATCH --error=/gpfs/mariana/home/svloor/Documents/ViT/run_vit_tiny_%j.err

source /gpfs/mariana/home/svloor/Documents/ViT/.venv311/bin/activate
cd /gpfs/mariana/home/svloor/Documents/ViT/
export PYTHONPATH=/gpfs/mariana/home/svloor/Documents/ViT/src:$PYTHONPATH
mkdir -p new_runs

OUT=new_runs/vit_tiny_fi.json

# Save fault-free logits
python -m src.cli --model vit_tiny --batch_size 100 --max_batches 50 save --logits

# Fault-free baseline accuracy
python -m src.cli --model vit_tiny --batch_size 100 --max_batches 50 --repeat 1 -o $OUT fi --faults 0

# QKV
for i in $(seq 0 11); do
    python -m src.cli --model vit_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers qkv --component qkv --block $i
done

# PROJ
for i in $(seq 0 11); do
    python -m src.cli --model vit_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers proj --component proj --block $i
done

# FC1
for i in $(seq 0 11); do
    python -m src.cli --model vit_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers fc1 --component fc1 --block $i
done

# FC2
for i in $(seq 0 11); do
    python -m src.cli --model vit_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers fc2 --component fc2 --block $i
done
