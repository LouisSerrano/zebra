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

python3 scripts/tokenizer/train.py --config-name=vqvae2d data.dataset_name=vorticity model.codebook_size=2048 model.num_codebooks=1 model.init_dim=128 model.num_groups=32 "model.layers=['residual', 'compress_space', 'residual', 'compress_space', 'residual']" training.smoothing=True training.max_steps=120000 
