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
Additionally, you need to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [FrEIA](https://github.com/vislearn/FrEIA).

### Datasets
Please follow the repositories of the [VOS](https://github.com/deeplearning-wisc/vos) and [STUD](https://github.com/deeplearning-wisc/stud) papers to obtain and prepare both publicly available inlier datasets as well as the pre-processed outlier datasets.
**Note:** After you have prepared the datasets and placed in your dataset directory, you need to specify the correct dataset folder path in the ```/path/to/Flow-Feature-Synthesis/detection/core/datasets/setup_datasets.py``` by changing the ```/path/to/``` to the correct path. 


## Training procedure
**Step 1:** First and foremost, make sure you are inside the project folder by running
```
cd /path/to/Flow-Feature-Synthesis/detection 
```
**Step 2:** You need to download the pre-trained RegNetX-4.0GF backbone from [here](https://drive.google.com/file/d/1WyE_OIpzV_0E_Y3KF4UVxIZJTSqB7cPO/view?usp=sharing) and place it in ```/path/to/Flow-Feature-Synthesis/detection/configs/regnetx_detectron2.pth```.


**Step 3:** You need to change the folder path where the dataset is placed in each of the below commands before you start the training process. 

**For training FFS with PASCAL-VOC as the inlier image dataset**
```
python train_net_gmm.py  
--dataset-dir /path/to/dataset/VOC/  
--num-gpus 4 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml  
--random-seed 0 
--resume True  
```
The trained model will be saved at ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/model_final.pth```.

**For training FFS with BDD100K as the inlier video dataset**
```
python train_net_gmm.py  
--dataset-dir /path/to/dataset/BDD100k_video/bdd100k/  
--num-gpus 4 
--config-file BDD100K/FFS_regnet.yaml  
--random-seed 0 
--resume True  
```
The trained model will be saved at ```/path/to/Flow-Feature-Synthesis/detection/data/configs/BDD100k/FFS_regnet/random_seed_0/model_final.pth```.

**For training FFS with Youtube-VIS as the inlier video dataset**
```
python train_net_gmm.py  
--dataset-dir /path/to/dataset/Youtube-VIS/  
--num-gpus 4 
--config-file VIS/FFS_regnet.yaml  
--random-seed 0 
--resume True  
``` 
The trained model will be saved at ```/path/to/Flow-Feature-Synthesis/detection/data/configs/VIS/FFS_regnet/random_seed_0/model_final.pth```


## Using Pre-trained FFS Models
If you would like to directly use the pre-trained FFS models instead of training a new instance of the FFS, please follow the below steps:
**Step 1:** You need to download the pre-trained FFS models for PASCAL-VOC, BDD100K Video and Youtube VIS datasets from [here](https://drive.google.com/drive/folders/1QGUn75onqWh6GUrmiPTCGP9o94PMHMeL?usp=share_link). Each of these models are trained with RegNetX as the backbone architecture. 
**Step 2:** Place them in the exact same folder as the folders where the trained models gets saved for the training procedure. 

## Inference procedure
Note:  The inference procedure is common irrespective of whether you use pre-trained models or trained the models following the training procedure mentioned before. 

**Evaluation with the FFS trained on PASCAL-VOC as the inlier dataset**
**Step 1:** First, the evaluation needs to be performed on the validation set of PASCAL-VOC as follows:

```
python apply_net.py  
--dataset-dir /path/to/dataset/VOC/
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
--dataset-dir /path/to/dataset/COCO/ 
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
--dataset-dir /path/to/dataset/OpenImages/  
--test-dataset openimages_ood_val 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml  
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```

**Step 3:** Finally, performance based on the evaluation metric such as FPR95 and AUROC can be computed as follows:

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
