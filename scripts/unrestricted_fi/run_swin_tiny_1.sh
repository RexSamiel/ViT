#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fi_swin_tiny_1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/gpfs/mariana/home/svloor/Documents/ViT/run_swin_tiny_1_%j.out
#SBATCH --error=/gpfs/mariana/home/svloor/Documents/ViT/run_swin_tiny_1_%j.err

source /gpfs/mariana/home/svloor/Documents/ViT/.venv311/bin/activate
cd /gpfs/mariana/home/svloor/Documents/ViT/
export PYTHONPATH=/gpfs/mariana/home/svloor/Documents/ViT/src:$PYTHONPATH
mkdir -p new_runs

OUT=new_runs/swin_tiny_fi.json

# Swin-tiny block layout:
#   layers.0: blocks 0-1
#   layers.1: blocks 0-1
#   layers.2: blocks 0-5
#   layers.3: blocks 0-1

# Save fault-free logits
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 save --logits

# Fault-free baseline accuracy
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1 -o $OUT fi --faults 0

# QKV
for stage in 0 1 3; do
    for block in 0 1; do
        python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers qkv --component qkv --layer_prefix "layers.${stage}.blocks.${block}"
    done
done
for block in 0 1 2 3 4 5; do
    python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers qkv --component qkv --layer_prefix "layers.2.blocks.${block}"
done

# PROJ
for stage in 0 1 3; do
    for block in 0 1; do
        python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers proj --component proj --layer_prefix "layers.${stage}.blocks.${block}"
    done
done
for block in 0 1 2 3 4 5; do
    python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers proj --component proj --layer_prefix "layers.2.blocks.${block}"
done
