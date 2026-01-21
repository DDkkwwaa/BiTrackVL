#!/bin/bash
#SBATCH -J pytorch
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH -o /share/home/u2304300116/lihui/HIPTrack_VLT_1765018150/HIPB_up_large/Large_log1.txt
export PYTHONPATH=/share/home/u2304300116/lihui/HIPTrack:$PYTHONPATH


source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate /share/home/u2415423004/.conda/envs/HIPTrack
cd /share/home/u2304300116/lihui/HIPTrack_VLT_1765018150/HIPB_up_large

python tracking/train.py --script hiptrack --config hiptrack_train_full --save_dir ./output --mode multiple --nproc_per_node 4