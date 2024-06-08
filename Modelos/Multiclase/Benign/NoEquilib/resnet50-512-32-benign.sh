#!/bin/bash


#SBATCH --job-name RN50-512-benign                # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

#SBATCH -c 10

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/hexecode/pt23_env

export TFHUB_CACHE_DIR=.


python pt-resnet-benign.py
