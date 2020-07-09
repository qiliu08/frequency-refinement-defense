
import numpy as np
from PIL import Image
import os
import copy
import cv2

import torch


def save_input_image(modified_im, im_name, folder_name='result_images', save_flag=True):
    """
    Discretizes 0-255 (real) image from 0-1 normalized image
    """
    modified_copy = copy.deepcopy(modified_im)[0]
    modified_copy = ((modified_copy * 0.5) + 0.5)
    modified_copy = modified_copy * 255
    # Box constraint
    modified_copy[modified_copy > 255] = 255
    modified_copy[modified_copy < 0] = 0
    modified_copy = modified_copy.transpose(1, 2, 0)
    modified_copy = modified_copy.astype('uint8')
    modified_copy = cv2.cvtColor(modified_copy, cv2.COLOR_RGB2BGR)
    if save_flag:
        save_image(modified_copy, im_name, folder_name)
    return modified_copy


def save_prediction_image(pred_out, im_name, folder_name='result_images'):
    """
    Saves the prediction of a segmentation model as a real image
    """
    # Disc. pred image
    pred_img = copy.deepcopy(pred_out)
    pred_img = pred_img * 255
    pred_img[pred_img > 127] = 255
    pred_img[pred_img <= 127] = 0
    save_image(pred_img, im_name, folder_name)


def save_image_difference(org_image, perturbed_image, im_name, folder_name='result_images'):
    """
    Finds the absolute difference between two images in terms of grayscale plaette
    """
    # Process images
    im1 = save_input_image(org_image, '', '', save_flag=False)
    im2 = save_input_image(perturbed_image, '', '', save_flag=False)
    # Find difference
    diff = np.abs(im1 - im2)
    # # calculate average
    avg = np.mean(diff)
    # max = np.amax(diff)
    # print('perturbation max: ', max)
    # Sum over channel
    diff = np.sum(diff, axis=2)
    # Normalize
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff = np.clip((diff - diff_min) / (diff_max - diff_min), 0, 1)
    # Enhance x 120, modify this according to your needs
    diff = diff*120
    diff = diff.astype('uint8')
    save_image(diff, im_name, folder_name)


def save_image(im_as_arr, im_name, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    image_name_with_path = folder_name + '/' + str(im_name) + '.png'
    # pred_img = Image.fromarray(im_as_arr)
    # print(im_as_arr.shape)
    #pred_img.save(image_name_with_path)
    cv2.imwrite(image_name_with_path, im_as_arr)


def load_model(path_to_model):
    """
    Loads pytorch model from disk
    """
    model = torch.load(path_to_model)
    #model = torch.load(path_to_model, map_location=lambda storage, loc: storage)
    return model

def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = torch.from_numpy(SR)
    GT = torch.from_numpy(GT)
    SR[SR > threshold] = 1
    SR[SR <= threshold] = 0
    # print(torch.sum(SR[SR > 0.5]))
    GT = GT == torch.max(GT)
    # print(torch.sum(GT[GT==1]))
    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)
    # Calculate pixel accuracy
    # correct_pixels = SR == GT
    # pixel_acc = np.sum(correct_pixels) / (correct_pixels.shape[0] * correct_pixels.shape[1])
    return JS

def get_PX_accuracy(SR,GT,threshold=0.5):
    SR = torch.from_numpy(SR)
    GT = torch.from_numpy(GT)
    #SR[SR > threshold] = 1
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def calculate_mask_similarity(mask1, mask2):
    """
    Calculates IOU and pixel accuracy between two masks
    """
    # Calculate IoU
    mask1[mask1 > 0.5] = 1
    mask1[mask1 <= 0.5] = 0
    intersection = mask1 * mask2
    union = mask1 + mask2
    # Update intersection 2s to 1
    union[union > 1] = 1
    iou = np.sum(intersection) / np.sum(union)

    # Calculate pixel accuracy
    correct_pixels = mask1 == mask2
    pixel_acc = np.sum(correct_pixels) / (correct_pixels.shape[2]*correct_pixels.shape[3])
    return (iou, pixel_acc)


def calculate_image_distance(im1, im2):
    """
    Calculates L2 and L_inf distance between two images
    """
    # Calculate L2 distance
    l2_dist = torch.dist(im1, im2, p=2).item()

    # Calculate Linf distance
    diff = torch.abs(im1 - im2)
    diff = torch.max(diff, dim=2)[0]  # 0-> item, 1-> pos
    linf_dist = torch.max(diff).item()
    return l2_dist, linf_dist

