# FFS: Normalizing Flow based Feature Synthesis for Outlier-Aware Object Detection
This repository provides the source code for the CVPR 2023 highlight [paper](https://arxiv.org/abs/2302.07106) by N. Kumar et al. The key contributions of the paper are as follows:
* Learns inlier features using invertible Normalizing Flow model within the object detection framework. 
* Generates synthetic outlier features in the inverse direction after random sampling from the latent space of the Flow model. 
* Regularizes the object detection framework to make it outlier-aware via discriminative training procedure that separates the energy surface of the synthesized outliers and inlier features. 
* Performs outlier-aware object detection on both image and video based datasets. 
# Key Diagram
Coming Soon

## Content
* [How to Cite](#citation)
* [Installation](#Installation)
  * [Package Requirements](#package-requirements)
  * [Datasets](#Datasets)
* [Training FFS from scratch](#training)
* [Using Pre-trained FFS Models](#pretrained-models)
* [Inference procedure](#inference)
* [Visualization of results](#visualization)

## How to Cite
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

## Installation

### Package Requirements
```
pip install -r requirements.txt
```
Additionally, you need to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [FrEIA](https://github.com/vislearn/FrEIA).

### Datasets
Please follow the repositories of the [VOS](https://github.com/deeplearning-wisc/vos) and [STUD](https://github.com/deeplearning-wisc/stud) papers to obtain and prepare both publicly available inlier datasets as well as the pre-processed outlier datasets.

**Note:** After you have prepared the datasets and placed in your dataset directory, you need to specify the correct dataset folder path in the ```/path/to/Flow-Feature-Synthesis/detection/core/datasets/setup_datasets.py``` by changing the ```/path/to/``` to the correct path. 


## Training FFS from scratch
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

**Step 2:** Place ```model_final.pth``` in the exact same folder where the trained models gets saved for the training procedure. Example: For the pre-trained model of PASCAL-VOC, you need to place the model at ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/model_final.pth```. 

**Note:** The path needs to be created if you directly use the pre-trained models instead of first performing the training from scratch. 

## Inference procedure
**Note:**  The inference procedure is common irrespective of whether you use pre-trained models or trained the models following the training procedure mentioned before. 

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

**Step 3:** Finally, the outlier detection performance based on the evaluation metric such as FPR95 and AUROC can be computed as follows:

For MS-COCO:
```
python voc_coco_plot.py
--name regnetx 
--thres xxx 
--energy 1 
--seed 0
```

**Note:** You can obtain the threshold value by looking at the name of the file ```probabilistic_scoring_res_odd_0.5959.txt``` at ```/path/to/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_xxxx.txt```.
For our pre-trained model with PASCAL-VOC dataset, the threshold should be ```0.5959``` and running the above script should result in the exact numbers as reported in our paper. 
 

For OpenImages:
```
python voc_openimage_plot.py
--name regnetx 
--thres xxx 
--energy 1 
--seed 0
```

**Evaluation with the FFS trained on BDD100K as the inlier video dataset**

**Step 1:** First, the evaluation needs to be performed on the validation set of BDD100K (Videos) as follows:

```
python apply_net.py  
--dataset-dir /path/to/dataset/BDD100k_video/bdd100k/
--test-dataset bdd_tracking_2k_val 
--config-file BDD100k/FFS_regnet.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```



**Evaluation with the FFS trained on Youtube-VIS as the inlier video dataset**

**Step 1:** First, the evaluation needs to be performed on the validation set of Youtube-VIS as follows:

```
python apply_net.py 
--dataset-dir /projects/p084/p_discoret/Youtube-VIS  
--test-dataset vis21_val 
--config-file VIS/FFS_regnet.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```



## Visualization of results


VOS vs FFS on nuImages and STUD vs FFS on OpenImages is not possible since ID and OD may not be mutually independent


For MS-COCO and OpenImage visualization on VOS trained on PASCAL-VOC dataset, the threshold for vos_visualize_voc = 16.189585. The procedure is a) run the indist, b) run the outdist, c) run python voc_coco_plot to print thresh from metric_utils, d) change line 131 RHS in inference_core.py to number of classes and then e) change the threshold to 16.189585 in line 97 of apply_net.py f) then final run outdist with visualize is true




For MS-COCO and OpenImage visualization on FFS trained on PASCAL-VOC dataset, the threshold for ffs_visualize_voc = 11.95079. The procedure is a) run the indist, b) run the outdist, c) run python voc_coco_plot to print thresh from metric_utils, d) change line 131 RHS in inference_core.py to number of classes and then e) change the threshold to 11.95079. in line 97 of apply_net.py f) then final run outdist with visualize is true (edited)




For nuImage visualization on STUD trained on BDD100K dataset, the threshold for stud_visualize_bdd = 3.490076. The procedure is: a) run the indist for STUD b) run the outdist for STUD c) run python bdd_coco.py with thresholds[cutoff] printed from metric_utils d) In src.engine.myvisualizer.py make sure line 114-115 is uncommented, line 112 is uncommented e) In src.engine.defaults.py line 768, change the threshold to the one found and change Line 685 RHS to number of classes i,e, 8 in case of BDD100k f) make sure in the outdist.sh script, you have first --config-file, --savefigdir (with path) and then --visualize



For MS-COCO visualization on STUD trained on Youtube-VIS dataset, the threshold for stud_visualize_bdd = 5.0098214. The procedure is: a) run the indist for STUD b) run the outdist for STUD c) run python vis_coco.py with thresholds[cutoff] printed from metric_utils d) In src.engine.myvisualizer.py make sure line 114-115 is uncommented, line 112 is uncommented e) In src.engine.defaults.py line 768, change the threshold to the one found and change Line 685 RHS to number of classes i,e, 40 in case of YoutubeVIS f) make sure in the outdist.sh script, you have first --config-file, --savefigdir (with path) and then --visualize



For MS-COCO visualization on FFS trained on Youtube-VIS dataset, the threshold is = 5.445534. The procedure is: a) run the indist b) run the outdist  c) run python vis_coco_plot.py with thresholds[cutoff] printed from metric_utils d) change line 131 RHS in inference_core.py to number of classes e) change the threshold to 5.445534 in line 97 of apply_net.py f) then final run outdist with visualize true



For nuImage visualization on FFS trained on BDD100K dataset, the threshold is = 5.7650447. The procedure is: a) run the indist b) run the outdist  c) run python vis_coco_plot.py with thresholds[cutoff] printed from metric_utils d) change line 131 RHS in inference_core.py to number of classes e) change the threshold to 5.7650447 in line 97 of apply_net.py f) then final run outdist with visualize true



