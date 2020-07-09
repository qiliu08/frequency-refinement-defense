# Frequency Refinement 

This repository contains the implementation of adversarial defense for deep learning biomedical segmentation models. This defense is proposed in a paper titled "Defending Deep Learning-based Biomedical Image Segmentation from Adversarial Attacks: A Low-cost Frequency Refinement Approach, " published in MICCAI-2020.

## Prerequisites
### Download pre-trained model 
Example model used in this repository can be downloaded from https://drive.google.com/file/d/19ZxTpbCm1pOuEDeldMB20tHvuKzfg9rJ/view?usp=sharing
### Download dataset
data/ folder contains a couple of images in ISIC dataset for testing purposes. The entire dataset can be found in https://challenge2018.isic-archive.com/task1

## Running the test
main.py includes biomedical segmentation-based adversarial examples generation and our Frequency Refinement defense processing. In adaptive_attack.py, We modify slightly original attack algorithm for fitting into our model. The original attack algorithm can be found in https://github.com/utkuozbulak/adaptive-segmentation-mask-attack#adaptive-segmentation-mask-attack.
you can simply run main.py to test our defense approach.
```
python main.py
```
