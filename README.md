# FFS: Normalizing Flow based Feature Synthesis for Outlier-Aware Object Detection
This repository provides the source code for the [CVPR 2023 highlight paper](https://arxiv.org/abs/2302.07106) titled **Normalizing Flow based Feature Synthesis for Outlier-Aware Object Detection** by [Nishant Kumar](https://tu-dresden.de/ing/informatik/smt/cgv/die-professur/mitarbeiter/nishant-kumar), [Siniša Šegvić](http://www.zemris.fer.hr/~ssegvic/), [Abouzar Eslami](https://scholar.google.de/citations?user=PmHOyT0AAAAJ&hl=en) and [Stefan Gumhold](https://tu-dresden.de/ing/informatik/smt/cgv/die-professur/inhaber-in). 

![GitHub Logo](/Flow-Feature-Synthesis/assets/Method.png)

**The key contributions of the paper are as follows:**
* Learns inlier features using invertible Normalizing Flow model within the object detection framework. 
* Generates synthetic outlier features in the inverse direction after random sampling from the latent space of the Flow model. 
* Regularizes the object detection framework to make it outlier-aware via discriminative training procedure that separates the energy surface of the synthesized outliers and inlier features. 
* Performs outlier-aware object detection on both image and video based datasets. Check out the [conference poster](/Flow-Feature-Synthesis/assets/FFS_poster.pdf).

# Video
[![Watch the video here](/Flow-Feature-Synthesis/assets/FFS_video_slides.png)](https://youtu.be/2LASy-H26lI)

# Advertisement
* Check out our AAAI 2024 work [QuantOD](https://github.com/taghikhah/QuantOD) on Outlier-aware Image Classification.
  
# Content
* [How to Cite](#how-to-cite)
* [Installation](#installation)
  * [Package Requirements](#package-requirements)
  * [Datasets](#datasets)
* [Training FFS from scratch](#training-ffs-from-scratch)
* [Using Pre-trained FFS Models](#using-pre-trained-ffs-models)
* [Inference procedure](#inference-procedure)
* [Visualization of results](#visualization-of-results)
* [License](#license)

# How to Cite
If you find this code or paper useful in your research, please consider citing our paper as follows:
```
@InProceedings{Kumar_2023_CVPR,
    author    = {Kumar, Nishant and \v{S}egvi\'c, Sini\v{s}a and Eslami, Abouzar and Gumhold, Stefan},
    title     = {Normalizing Flow Based Feature Synthesis for Outlier-Aware Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5156-5165}
}
```

# Installation

### Package Requirements
```
pip install -r requirements.txt
```
Additionally, you need to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [FrEIA](https://github.com/vislearn/FrEIA).

### Datasets
Please follow the repositories of the [VOS](https://github.com/deeplearning-wisc/vos) and [STUD](https://github.com/deeplearning-wisc/stud) papers to obtain and prepare both publicly available inlier datasets as well as the pre-processed outlier datasets.

**Note:** After you have prepared the datasets and placed in your dataset directory, you need to specify the correct dataset folder path in the ```/path/to/Flow-Feature-Synthesis/detection/core/datasets/setup_datasets.py``` by changing the ```/path/to/``` to the correct path wherever required. 


# Training FFS from scratch
**Step 1:** First and foremost, make sure you are inside the project folder by running
```
cd /path/to/Flow-Feature-Synthesis/detection 
```
**Step 2:** You could download the pre-trained RegNetX-4.0GF backbone from [here](https://drive.google.com/file/d/1WyE_OIpzV_0E_Y3KF4UVxIZJTSqB7cPO/view?usp=sharing) and place it in ```/path/to/Flow-Feature-Synthesis/detection/configs/regnetx_detectron2.pth```. You could already find this file at the given location in this repository. 


**Step 3:** You need to change the folder path where the dataset is placed in each of the below commands before you start the training process. 

**For training FFS with PASCAL-VOC as the inlier image dataset**
```
python train_net_gmm.py  --dataset-dir /path/to/dataset/VOC/  --num-gpus 4 --config-file VOC-Detection/faster-rcnn/regnetx.yaml  --random-seed 0 --resume True  
```
The trained model will be saved at ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/model_final.pth```.

**For training FFS with BDD100K as the inlier video dataset**
```
python train_net_gmm.py  --dataset-dir /path/to/dataset/BDD100k_video/bdd100k/  --num-gpus 4 --config-file BDD100K/FFS_regnet.yaml  --random-seed 0 --resume True  
```
The trained model will be saved at ```/path/to/Flow-Feature-Synthesis/detection/data/configs/BDD100k/FFS_regnet/random_seed_0/model_final.pth```.

**For training FFS with Youtube-VIS as the inlier video dataset**
```
python train_net_gmm.py  --dataset-dir /path/to/dataset/Youtube-VIS/  --num-gpus 4 --config-file VIS/FFS_regnet.yaml  --random-seed 0 --resume True  
``` 
The trained model will be saved at ```/path/to/Flow-Feature-Synthesis/detection/data/configs/VIS/FFS_regnet/random_seed_0/model_final.pth```

# [Optional] Training FFS with SGLD
You can also train our FFS model with the Stochastic Gradient based Langevin Dynamics approach as proposed in the paper. All you need to do is change the line 18 in ```/path/to/Flow-Feature-Synthesis/detection/modeling/plain_generalized_rcnn_logistic_gmm.py``` from ```from modeling.flow_generator import build_roi_heads``` to ```from modeling.flow_generator_sgld import build_roi_heads```. 

# Using Pre-trained FFS Models
If you would like to directly use the pre-trained FFS models instead of training a new instance of the FFS, please follow the below steps:

**Step 1:** You need to download the pre-trained FFS models for PASCAL-VOC, BDD100K Video and Youtube VIS datasets from [here](https://drive.google.com/drive/folders/1QGUn75onqWh6GUrmiPTCGP9o94PMHMeL?usp=share_link). Each of these models are trained with RegNetX as the backbone architecture. 

**Step 2:** Place ```model_final.pth``` in the exact same folder where the trained models gets saved for the training procedure. Example: For the pre-trained model of PASCAL-VOC, you need to place the model at ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/model_final.pth```. 

**Note:** The path needs to be created if you directly use the pre-trained models instead of first performing the training from scratch. Additionally, If you are using the weights from the pre-trained model, then please define the correct path for WEIGHTS in the config file as ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/model_final.pth```. If you do not perform this step, there may be a situation where detectron2 framework still uses initial weights from ```/path/to/Flow-Feature-Synthesis/detection/configs/regnetx_detectron2.pth``` and you may not get the desired results.

# Inference procedure
**Note:**  The inference procedure is common irrespective of whether you use pre-trained models or trained the models following the training procedure mentioned before. 

**Evaluation with the FFS trained on PASCAL-VOC as the inlier dataset**

**Step 1:** First, the evaluation needs to be performed on the validation set of PASCAL-VOC as follows:

```
python apply_net.py  --dataset-dir /path/to/dataset/VOC/ --test-dataset voc_custom_val --config-file VOC-Detection/faster-rcnn/regnetx.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

**Step 2:** Then, the evaluation needs to be performed on the validation set of outlier data. 

For MS-COCO:
```
python apply_net.py  --dataset-dir /path/to/dataset/COCO/ --test-dataset coco_ood_val --config-file VOC-Detection/faster-rcnn/regnetx.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

For OpenImages:
```
python apply_net.py  --dataset-dir /path/to/dataset/OpenImages/  --test-dataset openimages_ood_val --config-file VOC-Detection/faster-rcnn/regnetx.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

**Step 3:** Finally, the outlier detection performance based on the evaluation metric such as FPR95 and AUROC can be computed as follows:

For MS-COCO:
```
python voc_coco_plot.py --name regnetx --thres xxx --energy 1 --seed 0
```

**Note:** You can obtain the threshold value by looking at the name of the file ```probabilistic_scoring_res_odd_0.5959.txt``` at ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_xxxx.txt```.
For our pre-trained model with PASCAL-VOC dataset, the threshold should be ```0.5959``` and running the above script should result in the exact numbers as reported in our paper. 
 

For OpenImages:
```
python voc_openimage_plot.py --name regnetx --thres xxx --energy 1 --seed 0
```


**Evaluation with the FFS trained on BDD100K as the inlier video dataset**

**Step 1:** First, the evaluation needs to be performed on the validation set of BDD100K (Videos) as follows:

```
python apply_net.py  --dataset-dir /path/to/dataset/BDD100k_video/bdd100k/ --test-dataset bdd_tracking_2k_val --config-file BDD100k/FFS_regnet.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

**Step 2:** Then, the evaluation needs to be performed on the validation set of outlier data as follows:  

For MS-COCO:
```
python apply_net.py  --dataset-dir /path/to/dataset/COCO/ --test-dataset coco_2017_val_ood_wrt_bdd  --config-file BDD100k/FFS_regnet.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

For nuImages:
```
python apply_net.py  --dataset-dir /path/to/dataset/nuImages/ --test-dataset nu_bdd_ood  --config-file BDD100k/FFS_regnet.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```


**Step 3:** Now, compute the outlier detection performance. The threshold could be determined using the same procedure as mentioned before:

For MS-COCO:
```
python bdd_coco_plot.py --name regnetx --thres xxx --energy 1 --seed 0
```

For nuImages:
```
python bdd_nuImage_plot.py --name regnetx --thres xxx --energy 1 --seed 0
```



**Evaluation with the FFS trained on Youtube-VIS as the inlier video dataset**

**Step 1:** First, the evaluation needs to be performed on the validation set of Youtube-VIS as follows:

```
python apply_net.py --dataset-dir /path/to/dataset/Youtube-VIS  --test-dataset vis21_val --config-file VIS/FFS_regnet.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

**Step 2:** Then, the evaluation needs to be performed on the validation set of outlier data as follows:  

For MS-COCO
```
python apply_net.py  --dataset-dir /path/to/dataset/COCO/ --test-dataset vis_coco_ood  --config-file VIS/FFS_regnet.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```


For nuImages
```
python apply_net.py  --dataset-dir /path/to/dataset/nuImages/ --test-dataset nu_bdd_ood --config-file VIS/FFS_regnet.yaml  --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```


**Step 3:** Now, compute the outlier detection performance using metric scripts. The threshold could be determined using the same procedure as mentioned before:

For MS-COCO:
```
python vis_coco_plot.py --name regnetx --thres xxx --energy 1 --seed 0
```

For nuImages:
```
python vis_nuImage_plot.py --name regnetx --thres xxx --energy 1 --seed 0
```



# Visualization of results

To visualize the performance of FFS on the outlier datasets, you need to perform the following procedure:

**Step 1:** Run the evaluation scripts for the inliers, outliers and metrics i.e. Step 1, Step 2 and Step 3 respectively. 

**Step 2:** Note down the threshold score printed based on line 85 in ```metric_utils.py```. For our pre-trained models with inlier datasets as PASCAL-VOC, BDD100K(Video) and Youtube-VIS, the thresholds are 11.95079, 5.7650447 and 5.445534 respectively.

**Step 3:** Change the right hand side of the line 131 in ```inference_core.py``` to the number of inlier classes (e.g. for PASCAL-VOC, it should be 20 since there are 20 inlier classes in this dataset)

**Step 4:** Change the threshold in line 97 of ```apply_net.py``` to the value obtained in Step 2.

**Step 5:** Finally run the evaluation script for the outlier dataset with ```--visualize 1```. 


# License
This software is licensed under the MIT license. 





