#!/usr/bin/env bash

#SBATCH --account=project_2002016
#SBATCH --job-name=emb
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1

module load gcc/8.3.0 cuda/10.1.168

export DIR=/projappl/project_2002016
cd $DIR

conda activate transf

srun python3 ./sequence-labeler/embedder.py