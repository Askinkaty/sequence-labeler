#!/usr/bin/env bash

#SBATCH --account=project_2002016
#SBATCH --job-name=seq_ged
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1

module load gcc/8.3.0 cuda/10.1.168

export DIR=/projappl/project_2002016/sequence-labeler
cd $DIR

conda activate transf

srun python3 experiment.py ./conf/fcepublic.conf