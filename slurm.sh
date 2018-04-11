#!/bin/bash
#SBATCH --time=15:00:00   # walltime - this is one hour
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G   # memory per CPU core

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load defaultenv
module add cuda/8.0
module add cudnn/6.0_cuda-8.0
module add python/2/7
module add python-pytorch/0.2_python-2.7_gcc-5.3_cuda-8.0_cudnn-6.0
python train.py
