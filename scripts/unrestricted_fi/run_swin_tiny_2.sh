#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fi_swin_tiny_2
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/gpfs/mariana/home/svloor/Documents/ViT/run_swin_tiny_2_%j.out
#SBATCH --error=/gpfs/mariana/home/svloor/Documents/ViT/run_swin_tiny_2_%j.err

source /gpfs/mariana/home/svloor/Documents/ViT/.venv311/bin/activate
cd /gpfs/mariana/home/svloor/Documents/ViT/
export PYTHONPATH=/gpfs/mariana/home/svloor/Documents/ViT/src:$PYTHONPATH
mkdir -p new_runs

OUT=new_runs/swin_tiny_fi.json

# FC1
for stage in 0 1 3; do
    for block in 0 1; do
        python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers fc1 --component fc1 --layer_prefix "layers.${stage}.blocks.${block}"
    done
done
for block in 0 1 2 3 4 5; do
    python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers fc1 --component fc1 --layer_prefix "layers.2.blocks.${block}"
done

# FC2
for stage in 0 1 3; do
    for block in 0 1; do
        python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers fc2 --component fc2 --layer_prefix "layers.${stage}.blocks.${block}"
    done
done
for block in 0 1 2 3 4 5; do
    python -m src.cli --model swin_tiny --batch_size 100 --max_batches 50 --repeat 1000 -o $OUT fi --faults 1 --layers fc2 --component fc2 --layer_prefix "layers.2.blocks.${block}"
done
