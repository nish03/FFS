# FFS: Normalizing Flow based Feature Synthesis for Outlier-Aware Object Detection
This repository provides the source code for the CVPR 2023 highlight [paper](https://arxiv.org/abs/2302.07106) by N. Kumar et al. The key contributions of the paper are as follows:
* Learns inlier features using invertible Normalizing Flow model within the object detection framework. 
* Generates synthetic outlier features in the inverse direction after random sampling from the latent space of the Flow model. 
* Regularizes the object detection framework to make it outlier-aware via discriminative training procedure that separates the energy surface of the synthesized outliers and inlier features. 

## Content
* [Installation](#Installation)
  * [Package Requirements](#package-requirements)
  * [Datasets](#Datasets)
* [FFS on Images](#ffs-on-images)
  * [(Images) Pre-trained models](#images-pre-trained-models)
  * [(Images) Training](#images-training)
  * [(Images) Inference](#images-inference)
* [FFS on Video](#ffs-on-video) 
  * [(Video) Pre-trained models](#video-pre-trained-models)
  * [(Video) Training](#video-training)
  * [(Video) Inference](#video-inference)

## Installation

### Package Requirements
```
pip install -r requirements.txt
```
Please note that the package versions in the requirements text file is  Additionally, you need to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [FrEIA](https://github.com/vislearn/FrEIA).

### Datasets
We followed [VOS](https://github.com/deeplearning-wisc/vos) and [STUD](https://github.com/deeplearning-wisc/stud) papers to download and prepare both publicly available inlier datasets as well as the pre-processed outlier datasets. Please follow both these repositories for the dataset preparation. 

## FFS on Images

### (Images) Pre-trained models
### (Images) Training
### (Images) Evaluation

## FFS on Video

### (Video) Pre-trained models
### (Video) Training
### (Video) Evaluation
