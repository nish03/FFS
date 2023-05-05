#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10G  
#SBATCH --partition=alpha
module load modenv/hiera GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 Python/3.8.6 
source ~/python-environments/torchvision_env/bin/activate
python apply_net.py  --dataset-dir /projects/p084/p_discoret/nuImages/ --test-dataset nu_bdd_ood  --config-file VIS/stud_regnet.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
 
