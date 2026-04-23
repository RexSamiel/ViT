#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=fi_test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/gpfs/mariana/home/svloor/Documents/ViT/test_run_%j.out
#SBATCH --error=/gpfs/mariana/home/svloor/Documents/ViT/test_run_%j.err

set -e

source /gpfs/mariana/home/svloor/Documents/ViT/.venv311/bin/activate
cd /gpfs/mariana/home/svloor/Documents/ViT/
export PYTHONPATH=/gpfs/mariana/home/svloor/Documents/ViT/src:$PYTHONPATH
mkdir -p new_runs

OUT=new_runs/test_fi.json

# 1 batch, 5 repeats, one block each model — just verify pipeline works end to end

python -m src.cli --model vit_tiny --max_batches 1 save --logits
python -m src.cli --model vit_tiny --batch_size 100 --max_batches 1 --repeat 5 -o $OUT fi --faults 1 --layers qkv --component qkv --block 0

python -m src.cli --model deit_tiny --max_batches 1 save --logits
python -m src.cli --model deit_tiny --batch_size 100 --max_batches 1 --repeat 5 -o $OUT fi --faults 1 --layers qkv --component qkv --block 0

python -m src.cli --model swin_tiny --max_batches 1 save --logits
python -m src.cli --model swin_tiny --batch_size 100 --max_batches 1 --repeat 5 -o $OUT fi --faults 1 --layers qkv --component qkv

echo "Test complete. Check new_runs/test_fi.json"
