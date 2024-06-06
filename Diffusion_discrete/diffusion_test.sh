#!/bin/bash
#PBS -l select=1:ncpus=4:mem=24gb
#PBS -l walltime=12:00:00
#PBS -N Diffusion_test
#PBS -koed

# module load anaconda3/personal

source $HOME/anaconda3/bin/activate
source activate si_training_3  # change the name to the goal environment

# module load tools/prod
# module load intel/2021b

cd $PBS_O_WORKDIR

cd /rds/general/user/qy320/home/diffusion
python3 -u train.py --training_selection 22 --layer_dim 1024 --num_demos 2000 --num_epochs 20000

# qsub diffusion_test.sh
# qstat
# qdel 9373759