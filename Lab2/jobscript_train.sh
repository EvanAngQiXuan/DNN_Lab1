#!/bin/bash
#SBATCH --job-name=disc_train
#SBATCH --partition=digitallabshort
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=train_%j.log

# Usage: one job per hyperparameter setting, arguments are passed to train.py
#   sbatch jobscript_train.sh --run_name adam_1e-3     --optimizer adam --lr 1e-3 --batch_size 8  --epochs 50
#   sbatch jobscript_train.sh --run_name adam_1e-4     --optimizer adam --lr 1e-4 --batch_size 8  --epochs 50
#   sbatch jobscript_train.sh --run_name sgd_1e-2_cos  --optimizer sgd  --lr 1e-2 --batch_size 16 --epochs 50
# Requires ../Messidor_Processed (run jobscript.sh / proc_messidor.py first)

# start environments and load existing modules
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python3 train.py "$@"
