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

run_name=tokenizer_heat2_emb256_dim64
python3 scripts/llama/pretrain.py data.dataset_name=heat2 data.tokenizer_path="/mnt/home/lserrano/zebra/outputs/${run_name}/last.ckpt" 
