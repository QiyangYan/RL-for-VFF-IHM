#!/bin/bash
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N Diffusion_2
#PBS -koed

# module load anaconda3/personal

source $HOME/anaconda3/bin/activate
source activate si_training_3  # change the name to the goal environment

# module load tools/prod
# module load intel/2021b

cd $PBS_O_WORKDIR

cd /rds/general/user/qy320/home/diffusion
python3 -u train.py --training_selection 2 --layer_dim 1024 --num_epochs 20000 --batch_size 1024

# specify these in the training file in case modification is needed

# qsub diffusion_training_2.sh
# qstat
# qdel 9373759