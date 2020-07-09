
import sys
import torch.nn as nn
import numpy
import copy
import random
random.seed(4)
import torch
from torch.autograd import Variable
from helper_functions import (save_prediction_image,
                              save_input_image,
                              save_image_difference,
                              calculate_mask_similarity,
                              calculate_image_distance)


class AdaptiveSegmentationMaskAttack:
    def __init__(self, device_id, model, tau, beta):
        self.device_id = device_id
        self.model = model
        self.tau = tau
        self.beta = beta

    def update_perturbation_multiplier(self, beta, tau, iou):
        return beta * iou + tau

    def calculate_l2_loss(self, x, y):
        loss = (x - y)**2
        for a in reversed(range(1, loss.dim())):
            loss = loss.sum(a, keepdim=False)
        loss = loss.sum()
        return loss

    def calculate_pred_loss(self, target_mask, pred_out, model_output):
        loss = 0

        out_channel = model_output[0][0]  # 1x1x224x224 -> 224*224
        opt_background_mask = target_mask.clone()
        opt_background_mask[opt_background_mask != 0] = self.temporary_class_id
        opt_background_mask[opt_background_mask == 0] = 1
        opt_background_mask[opt_background_mask == self.temporary_class_id] = 0

        pred_central_class_mask = pred_out[0].clone()
        pred_central_class_mask[pred_central_class_mask != 0] = self.temporary_class_id
        pred_central_class_mask[pred_central_class_mask == 0] = 0
        pred_central_class_mask[pred_central_class_mask == self.temporary_class_id] = 1

        central_class_to_background_loss = torch.sum(out_channel * opt_background_mask * pred_central_class_mask)
        opt_central_class_mask = target_mask.clone()
        opt_central_class_mask[opt_central_class_mask != 1] = self.temporary_class_id
        opt_central_class_mask[opt_central_class_mask == 1] = 1
        opt_central_class_mask[opt_central_class_mask == self.temporary_class_id] = 0

        pred_background_mask = pred_out[0].clone()
        pred_background_mask[pred_background_mask != 1] = self.temporary_class_id
        pred_background_mask[pred_background_mask == 1] = 0
        pred_background_mask[pred_background_mask == self.temporary_class_id] = 1

        background_to_central_class_loss = torch.sum(out_channel * opt_central_class_mask * pred_background_mask)
        loss = central_class_to_background_loss - background_to_central_class_loss
        return loss

    def perform_attack(self, image_name, input_image, org_mask, target_mask,  
                       total_iter=2501, save_samples=True, save_path='adv_results/', verbose=True):
        if save_samples:
            # Save masks
            save_prediction_image(org_mask.numpy(), image_name + '_original_mask', save_path+'mask_diff')
            save_prediction_image(target_mask.numpy(), 'target_mask', save_path)
        # Unique classes are needed to simplify prediction loss
        # Have a look at calculate_pred_loss to see where this is used
        self. temporary_class_id = random.randint(0, 999)

        # Assume there is no overlapping part for the first iteration
        pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, 0)
        # Get a copy of target mask to use it for stats
        org_mask_numpy = copy.deepcopy(org_mask).numpy()
        # target_mask_numpy = copy.deepcopy(target_mask).numpy()
        target_mask_numpy = target_mask.clone().numpy()
        # Target mask
        target_mask = (target_mask).float().cuda(self.device_id)

        # Image to perform the attack on
        image_to_optimize = input_image.clone()
        # Copied version of image for l2 dist
        org_im_copy = image_to_optimize.clone().cpu().cuda(self.device_id)
        for single_iter in range(total_iter):
            # Put in variable to get grads later on
            image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)

            # Forward pass
            out = self.model(image_to_optimize)

            numpy.set_printoptions(threshold=sys.maxsize)
            torch.set_printoptions(threshold=sys.maxsize)

            out = torch.transpose(out, 2, 1)

            out = out.reshape((1, 1, 224, 224))
            pred_out = out.clone()
            pre_out_np = pred_out.detach().cpu().numpy()
            ori_iou, pixel_acc = calculate_mask_similarity(pre_out_np, org_mask_numpy)
            # pred_out data process
            pred_out[pred_out > 0.5] = 1
            pred_out[pred_out <= 0.5] = 0
            pred_out = torch.squeeze(pred_out, 1).float()

            # L2 Loss
            l2_loss = self.calculate_l2_loss(org_im_copy, image_to_optimize)
            # Prediction loss
            pred_loss = self.calculate_pred_loss(target_mask, pred_out, out)

            # Total loss
            out_grad = torch.sum(pred_loss + l2_loss)
            # Backward pass
            out_grad.backward()

            # Add perturbation to image to optimize

            perturbed_im = image_to_optimize.data - (image_to_optimize.grad * pert_mul)
            # Do another forward pass to calculate new pert_mul
            perturbed_im_out = self.model(perturbed_im)
            perturbed_im_out = torch.transpose(perturbed_im_out, 2, 1)
            perturbed_im_out = perturbed_im_out.reshape((1, 1, 224, 224))
            # Discretize perturbed image to calculate stats

            perturbed_im_pred = perturbed_im_out.detach().cpu().numpy()
            # Calculate performance of the attack
            # Similarities
            iou, pixel_acc = calculate_mask_similarity(perturbed_im_pred, target_mask_numpy)

            # Distances
            l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
            # Update perturbation multiplier
            pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, iou)

            # Update image to optimize and ensure boxt constraint
            image_to_optimize = perturbed_im.data.clamp_(-1, 1)

            if single_iter % 500 == 0:
                if verbose:
                    print('Iter:', single_iter, '\tIOU (targeted attack):', iou, 
                          '\tIOU (original):', ori_iou,
                          '\n\t\tL2 Dist:', l2_dist,
                          '\tL_inf dist:', linf_dist)
        if save_samples:
            save_prediction_image(pred_out.cpu().detach().numpy()[0], image_name + '_iter_' + str(single_iter), save_path+'mask_diff')
            save_input_image(image_to_optimize.data.cpu().detach().numpy(),image_name + '_iter_' + str(single_iter), save_path+'modified_image')
            save_image_difference(image_to_optimize.data.cpu().detach().numpy(), org_im_copy.data.cpu().detach().numpy(), image_name +'_iter_' + str(single_iter), save_path+'added_perturbation')