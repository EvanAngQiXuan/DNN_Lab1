#!/bin/bash
#SBATCH --job-name=messidor_prep
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=messidor_prep_%j.log # Standard output and error log (%j = Job ID)

# Purge any default modules and load Python
module purge
module load Python/3.10.4-GCCcore-11.3.0

# Install required dependencies to your local user directory if not already installed
pip install --user --upgrade scipy opencv-python scikit-learn tqdm

# Execute the processing script
python3 proc_messidor.py