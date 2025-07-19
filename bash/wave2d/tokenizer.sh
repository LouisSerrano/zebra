#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/zebra/bin/activate 

python3 scripts/tokenizer/train.py --config-name=vqvae2d data.dataset_name=wave2d model.pad_mode="zeros" model.codebook_size=2048 training.max_steps=250000
