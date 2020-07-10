
import torch
import cv2
import numpy as np
import copy
import os
import argparse
from model import Net
import dataset
from helper_functions import save_prediction_image, calculate_mask_similarity
from frequency_refinement import FrequencyRefinement
from adaptive_attack import AdaptiveSegmentationMaskAttack


def main():

    parser = argparse.ArgumentParser()
    # attack parameters
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for attack')
    parser.add_argument('--tau', type=float, default=1e-7, help='attack parameter: tau')
    parser.add_argument('--beta', type=float, default=1e-6, help='attack parameter: beta')
    parser.add_argument('--total_iter', type=int, default=1001, help='total iteration for ASMA algorithm')    
    # defense parameters   
    parser.add_argument('--PD_band', type=int, default=30, help='priority defense (PD) band')
    parser.add_argument('--AC_band', type=int, default=5, help='accuracy compensation (AC) band')
    parser.add_argument('--GD_band', type=int, default=60, help='global defense (GD) band') 
    # GPU parameter
    parser.add_argument('--DEVICE_ID', type=int, default=0, help='device ID of GPU')
    # model parameter
    parser.add_argument('--model_path',default='models/lesions.pth', type=str, help='Path to dataset')

    args = parser.parse_args()
    
    lesionData = dataset.LesionData()
    loader = lesionData.getDataLoader(batch_size=args.batch_size)
    
    # Load model
    net = Net().cuda()
    net.load(args.model_path)
    net.eval()
    net.cuda(args.DEVICE_ID)
   
    # read images
    iterate = 0
    im_name = [0] * len(loader)
    org_img = [0] * len(loader)
    org_mask = [0] * len(loader)
    for images, labels, fnames in loader:
        labels = torch.squeeze(labels)
        labels = labels.reshape((224, 224))
        if iterate < len(loader)-1:
            org_img[iterate], org_mask[iterate], im_name[iterate] = images, labels, str(fnames)[2: -3]
        else:
            # the mask of last image as target mask for attack
            _, tar_mask, _ = images, labels, str(fnames)[2: -3]
        iterate = iterate + 1

    
    print('----------------------Adaptive Segmentation Mask Attack------------------------')
    adaptive_attack = AdaptiveSegmentationMaskAttack(args.DEVICE_ID, net, args.tau, args.beta)
    for i in range(0, len(loader)-1):
        print('The ' + str(i+1) + 'th image')
        adaptive_attack.perform_attack(im_name[i], org_img[i], org_mask[i], tar_mask, total_iter = args.total_iter)
    
    
    print('----------------------Frequency Refinement Defense------------------------')
    FR_defense = FrequencyRefinement(input_preprocessing_type='padding', PD_band = args.PD_band, AC_band = args.AC_band, GD_band = args.GD_band)    
    for i in range(0, len(loader) - 1):
        print('The ' + str(i+1) + 'th image')
        AE_image = cv2.cvtColor(cv2.imread('adv_results/modified_image/' + im_name[i] + '_iter_' + str(args.total_iter-1)+'.png'), cv2.COLOR_BGR2RGB)
        AE_image = AE_image.reshape((1, AE_image.shape[0], AE_image.shape[1], AE_image.shape[2]))        
        mitigated_FD_image = FR_defense.refinement_processing(AE_image)

        # input preprocessing
        mitigated_FD_image = mitigated_FD_image / 255
        mitigated_FD_image = np.clip(np.float32(mitigated_FD_image), 0.0, 1.0)        
        mitigated_FD_image = mitigated_FD_image[0].transpose(2,0,1)
        mitigated_FD_image = (mitigated_FD_image - 0.5) / 0.5
        mitigated_FD_image = torch.from_numpy(mitigated_FD_image).float().unsqueeze(0).cuda()
        
        # validation
        out = net(mitigated_FD_image)
        out = torch.transpose(out, 2, 1)
        out = out.reshape((1, 1, 224, 224))
        perturbed_im_pred = out.detach().cpu().numpy()
        target_mask_numpy = copy.deepcopy(org_mask[i]).numpy()
                   
        recovered_iou, _ = calculate_mask_similarity(perturbed_im_pred, target_mask_numpy)
        print('recovered IOU',recovered_iou)
        
        # save recovered mask
        saveout = out.clone()
        saveout[saveout > 0.5] = 1
        saveout[saveout <= 0.5] = 0
        saveout = torch.squeeze(saveout, 1).float()
        if not os.path.exists('adv_results/FR_recover_mask/'):
            os.makedirs('adv_results/FR_recover_mask/')
        save_prediction_image(saveout.cpu().detach().numpy()[0], im_name[i], 'adv_results/FR_recover_mask')

   
    
if __name__ == '__main__':
    main()   





    
   
