#!/bin/bash


#SBATCH --job-name Prueba                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/hexecode/pt23_env

export TFHUB_CACHE_DIR=.


python pytorch_test.py
