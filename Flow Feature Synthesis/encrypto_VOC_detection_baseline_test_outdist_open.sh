#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G  
#SBATCH --partition=alpha
module load modenv/hiera GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 Python/3.8.6 
source ~/python-environments/torchvision_env/bin/activate
python apply_net.py  --dataset-dir /projects/p084/p_discoret/OpenImages/  --test-dataset openimages_ood_val --config-file VOC-Detection/faster-rcnn/regnetx.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 1
 
