# Frequency Refinement 

This repository contains the implementation of adversarial defense for deep learning biomedical segmentation models. This defense is proposed in our paper titled "Defending Deep Learning-based Biomedical Image Segmentation from Adversarial Attacks: A Low-cost Frequency Refinement Approach, " published in MICCAI-2020.

## Prerequisites
### Download pre-trained model 
Example model used in this repository can be downloaded from https://drive.google.com/file/d/19ZxTpbCm1pOuEDeldMB20tHvuKzfg9rJ/view?usp=sharing
### Download dataset (optional)
data/ folder contains a couple of images in ISIC dataset for testing purposes. The entire dataset can be found in https://challenge2018.isic-archive.com/task1

## Running the test
main.py includes biomedical segmentation-based adversarial examples generation and our Frequency Refinement defense processing. In adaptive_attack.py, we slightly modify the original attack algorithm for fitting into our model. The original adversarial attack algorithm can be found in https://github.com/utkuozbulak/adaptive-segmentation-mask-attack#adaptive-segmentation-mask-attack.
you can run main.py to test our defense approach.
```
python main.py
```
## Requirements
```
python=3.7.6
pytorch=1.3.1
torchvision = 0.4.2
```

## Citation
If you find this project is useful for your research, consider citing our paper. 
