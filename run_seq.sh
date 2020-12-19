#!/usr/bin/env bash

#SBATCH --account=project_2002016
#SBATCH --job-name=seq_ged
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1

module load gcc/8.3.0 cuda/10.1.168

export DIR=/projappl/project_2002016
cd $DIR

conda activate transf

srun python3 ./sequence-labeler/experiment.py ./sequence-labeler/conf/gram_det.conf