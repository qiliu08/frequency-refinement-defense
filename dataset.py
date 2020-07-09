import glob
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import _get_inverse_affine_matrix

import math
import matplotlib.pyplot as plt

IMAGE_HT = 224
IMAGE_WD = 224

class LesionDataset(data.Dataset):
    highlighted_color = np.array([255, 255, 255])

    input_processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, x, y, imgnos, input_preprocessor, augment=False):
        super().__init__()
        self.imgnos = imgnos
        self.y = y
        self.x = x  
        self.input_preprocessor = input_preprocessor
        self.augment = augment

    def imread(self, file_name):
        return cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)

    def labelcvt(self, img):
        gt_bg = np.all(img == LesionDataset.highlighted_color, axis=2)
        gt_bg = np.expand_dims(gt_bg, 2)

        class1 = np.zeros(gt_bg.shape, dtype=np.float32)
        class1[gt_bg] = 1.
        return class1.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = [self.imread(self.x[idx])] + [self.imread(y) for y in self.y[idx]]
        x = self.input_preprocessor(item[0])
        if len(item) > 2:
            y = np.dstack([self.labelcvt(tgt) for tgt in item[1:]]).squeeze()
        else:
            y = self.labelcvt(item[1])
        return x, y, self.imgnos[idx]


class LesionData(object):
    def __init__(self):
         self.table = {}

    def getShape(self, imgno):
        return self.table[imgno][0]

    def getROI(self, imgno):
        return self.table[imgno][1]

    def getImgNos(self):
        return list(self.table.keys())

    def getDataLoader(self, batch_size):
        imgnos = []
        for filename in glob.glob("data/ISIC_Mask/*_mask.png"):
            imgnos.append(filename[filename.rfind('/')+1: filename.rfind('_')])

        eval_x = ['data/ISIC_Input/{}.png'.format(n) for n in imgnos]
        eval_y = [['data/ISIC_Mask/{}_mask.png'.format(n)] for n in imgnos]

        eval_dataset = LesionDataset(eval_x, eval_y, imgnos, LesionDataset.input_processor, augment=False)
        loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        return loader




