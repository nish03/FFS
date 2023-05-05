# FFS: Normalizing Flow based Feature Synthesis for Outlier-Aware Object Detection
This repository provides the source code for the CVPR 2023 highlight [paper](https://arxiv.org/abs/2302.07106) by N. Kumar et al. The key contributions of the paper are as follows:
* Learns inlier features using invertible Normalizing Flow model within the object detection framework. 
* Generates synthetic outlier features in the inverse direction after random sampling from the latent space of the Flow model. 
* Regularizes the object detection framework to make it outlier-aware via discriminative training procedure that separates the energy surface of the synthesized outliers and inlier features. 

## Content
* [Installation](#Installation)
  * [Package Requirements](#package-requirements)
  * [Datasets](#Datasets)
* [Using Pre-trained Models](#pretrained-models)
* [Training from scratch](#training)
* [Inference procedure](#inference)
* [Visualization of results](#visualization)
* [How to Cite](#citation)


## Installation

### Package Requirements
```
pip install -r requirements.txt
```
Please note that the package versions in the requirements text file is  Additionally, you need to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [FrEIA](https://github.com/vislearn/FrEIA).

### Datasets
We followed [VOS](https://github.com/deeplearning-wisc/vos) and [STUD](https://github.com/deeplearning-wisc/stud) papers to obtain both publicly available inlier datasets as well as the pre-processed outlier datasets. Please follow both these repositories for the dataset preparation. 

## Using Pre-trained Models

## Training from scratch
First and foremost, make sure you are inside the project folder by running
```
cd Flow_Feature_Synthesis
```
Secondly, you need to change the folder path where the dataset is placed in each of the below command before you start the training

**For training FFS with PASCAL-VOC as the inlier image dataset**
```
python train_net_gmm.py  
--dataset-dir /path/to/dataset/dir/VOC/  
--num-gpus 4 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml  
--random-seed 0 
--resume True  
```
**For training FFS with BDD100K as the inlier video dataset**
```
python train_net_gmm.py  
--dataset-dir /path/to/dataset/dir/BDD100k_video/bdd100k/  
--num-gpus 4 
--config-file BDD100K/stud_regnet.yaml  
--random-seed 0 
--resume True  
```
**For training FFS with Youtube-VIS as the inlier video dataset**
```
python train_net_gmm.py  
--dataset-dir /path/to/dataset/dir/Youtube-VIS/  
--num-gpus 4 
--config-file VIS/stud_regnet.yaml  
--random-seed 0 
--resume True  
``` 

## Inference procedure
**Evaluation with the FFS trained on PASCAL-VOC as the inlier dataset**

**Step 1:** First, the evaluation needs to be performed on the validation set of PASCAL-VOC as follows:
```
python apply_net.py  
--dataset-dir /path/to/dataset/dir/VOC/
--test-dataset voc_custom_val 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
**Step 2:** Then, the evaluation needs to be performed on the validation set of outlier data. 

For MS-COCO:
```
python apply_net.py  
--dataset-dir /path/to/dataset/dir/COCO/ 
--test-dataset coco_ood_val 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml  
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```

For OpenImages:
```
python apply_net.py  
--dataset-dir /path/to/dataset/dir/OpenImages/  
--test-dataset openimages_ood_val 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml  
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```

** Step 3:** Finally, performance based on the evaluation metric such as FPR95 and AUROC can be computed as follows:

For MS-COCO:
```
python voc_coco_plot.py
--name vos 
--thres xxx 
--energy 1 
--seed 0
```

For OpenImages:
```
python voc_openimage_plot.py
--name vos 
--thres xxx 
--energy 1 
--seed 0
```

## Visualization of results

## How to Cite
