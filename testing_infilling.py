#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 08 10:10:37 2024

@author: Arthur C.

Test the neural network to produce extra imgs betwen the real ones (no replacement)
Using the t parameter for positioning.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pydicom
import os
import argparse
import copy

from tqdm import tqdm

# Own codes
from libs.models import Gen, Net_t
from libs.utilities import load_model, makedir
from libs.dataset import scale, de_scale, zscore, de_zscore
from skimage.filters import threshold_otsu

#%%
def crop_imgs(img_1, img_2, img_3, cropped_dim, laterality, view, step, machine):
    
    #Get image shape
    h, w = img_1.shape
     
    # Margin to centralize cropped region on the middle of breast if view = CC
    margin = (h - cropped_dim[0])//2
    
    # ATTENTION - Make sure that the images are not all fliped to the right side, different equipment has different protocol
    # Hologic: Images are always on the right side. Siemens: Left breasts are displayed at the left side

    # Crop according to view and laterality
    if view == 'CC':
        if laterality == 'R' or machine == 'Hologic':
            img_1 = img_1[margin:cropped_dim[0]+margin,-cropped_dim[1]:]
            img_2 = img_2[margin:cropped_dim[0]+margin,-cropped_dim[1]:]
            img_3 = img_3[margin:cropped_dim[0]+margin,-cropped_dim[1]:]
        else: 
            img_1 = img_1[margin:cropped_dim[0]+margin,:cropped_dim[1]]
            img_2 = img_2[margin:cropped_dim[0]+margin,:cropped_dim[1]]
            img_3 = img_3[margin:cropped_dim[0]+margin,:cropped_dim[1]]
    elif laterality == 'R':
        img_1 = img_1[step:cropped_dim[0]+step,-cropped_dim[1]:]
        img_2 = img_2[step:cropped_dim[0]+step,-cropped_dim[1]:]
        img_3 = img_3[step:cropped_dim[0]+step,-cropped_dim[1]:]
    elif machine == 'Siemens':
        img_1 = img_1[step:cropped_dim[0]+step,:cropped_dim[1]]
        img_2 = img_2[step:cropped_dim[0]+step,:cropped_dim[1]]
        img_3 = img_3[step:cropped_dim[0]+step,:cropped_dim[1]] 
    elif step == 0:
        img_1 = img_1[-cropped_dim[0]:,-cropped_dim[1]:]
        img_2 = img_2[-cropped_dim[0]:,-cropped_dim[1]:]
        img_3 = img_3[-cropped_dim[0]:,-cropped_dim[1]:] 
    else:
        img_1 = img_1[-step-cropped_dim[0]:-step,-cropped_dim[1]:]
        img_2 = img_2[-step-cropped_dim[0]:-step,-cropped_dim[1]:]
        img_3 = img_3[-step-cropped_dim[0]:-step,-cropped_dim[1]:]
    
    return img_1, img_2, img_3
         
def model_forward(model, img_1, img_3, maxV, normalization, angle = 0, useTuningP=False, props=None):
    
    # Change model to eval
    model.eval()
    
    if normalization == 'scale':
        # Normalize image
        img_1 = scale(img_1.astype('float32'), vmin=0, vmax = maxV)
        img_3 = scale(img_3.astype('float32'), vmin=0, vmax = maxV)
    elif normalization == 'zscore':
        # Z-score standardization
        img_1 = zscore(img_1.astype('float32'), mean = props['mean'], std = props['std'])
        img_3 = zscore(img_3.astype('float32'), mean = props['mean'], std = props['std'])   

    # Allocate memory to speed up the for loop 
    interp_img_1 = np.empty_like(img_1)
    interp_img_2 = np.empty_like(img_1)
    
    # Forward through the model
    with torch.no_grad():
        if useTuningP:
            interp_img = model(torch.from_numpy(img_1).unsqueeze(0).unsqueeze(0).to(device),
                               torch.from_numpy(img_3).unsqueeze(0).unsqueeze(0).to(device),
                               torch.Tensor([angle]).to(device))
        else:
            interp_img_1 = model(torch.from_numpy(img_1).unsqueeze(0).unsqueeze(0).to(device),
                               torch.from_numpy(img_3).unsqueeze(0).unsqueeze(0).to(device),
                               t = 0.25)
            interp_img_2 = model(torch.from_numpy(img_1).unsqueeze(0).unsqueeze(0).to(device),
                               torch.from_numpy(img_3).unsqueeze(0).unsqueeze(0).to(device),
                               t = 0.75)
    
    # Get from GPU
    interp_img_1 = interp_img_1.squeeze().squeeze().to('cpu').numpy()
    interp_img_2 = interp_img_2.squeeze().squeeze().to('cpu').numpy()
    
    if normalization == 'scale':
        # Normalize image (Inv)
        interp_img_1 = de_scale(interp_img_1, vmax = maxV)
        interp_img_2 = de_scale(interp_img_2, vmax = maxV)
    elif normalization == 'zscore':
        # Z-score image back
        interp_img_1 = de_zscore(interp_img_1,mean = props['mean'], std = props['std'])
        interp_img_2= de_zscore(interp_img_2,mean = props['mean'], std = props['std'])
        
    interp_img_1_min = interp_img_1.min()
    interp_img_2_min = interp_img_2.min()
    
    if interp_img_1_min < 0:
        interp_img_1 += abs(interp_img_1_min)
    if interp_img_2_min < 0:
        interp_img_2 += abs(interp_img_2_min)
    
    return interp_img_1, interp_img_2

def correct_intensity(img, img1, img3, maxV):
    
    # Processed images
    maskBreast = img > threshold_otsu(img) 
    mean_img1 = img1[img1 > threshold_otsu(img1)].mean()
    mean_img2 = img3[img3 > threshold_otsu(img3)].mean()

    factor_b = img[~maskBreast].mean() # Brightness factor. In this case, equal to the offset in the bg.
    c_img = img - factor_b
    factor_c = c_img[maskBreast].mean()/np.mean([mean_img1,mean_img2]) # Contrast factor
    c_img = c_img/factor_c
    c_img = np.clip(c_img,0,maxV)
    c_img[~maskBreast] = 0
    
    return c_img

def test(model, path_data, path2write, partition, subset, machine, crop_flag, cropped_dim, correct_intensity_flag, loss, epoch, nPv, angR, maxV, normalization, useTuningP=False, applyFlatFielding=False, props=None):
        
    test_exams_path = os.listdir(os.path.join(path_data,subset))
    
    applyFlatFielding = False
    applyMeanFactor = False
    step = 0
    if applyFlatFielding:
        calibration_map = np.load(os.path.join(path_data,'SiemensProjCalibrationMap_{}_step_{}.npy'.format('L',step)))
        mean_factor = np.load(os.path.join(path_data,'SiemensProjMeanFactor.npy'))
    
    # Angles corresponding to the acquisition geometry and equipment
    angles = np.linspace(-angR/2,angR/2,nPv+nPv-1)
    
    for exam_folder in test_exams_path:
          
        file_names = sorted([str(item) for item in os.listdir(os.path.join(path_data,subset,exam_folder))],key=lambda k: (len(k), k))
        folder_name = os.path.join(path2write,partition,normalization,'Infilling','DBT_PV_INTERPOLATION_DL_RRIN_{}_Epoch_{}/'.format(typ,epoch),subset,exam_folder)
        
        # Create output dir (if needed)
        makedir(folder_name)
        
        for i in tqdm(range(0,nPv-2,2)):
            print('input projections: {} and {}. Target angle: {}.'.format(i,i+1,angles[2*i+1]))
            skip_crop_flag = False
            
            # Read dicom image
            dcmH1 = pydicom.dcmread(os.path.join(path_data,subset,exam_folder,file_names[i]))
            dcmH2 = pydicom.dcmread(os.path.join(path_data,subset,exam_folder,file_names[i+1]))
            dcmH3 = pydicom.dcmread(os.path.join(path_data,subset,exam_folder,file_names[i+2]))
            
            # Read dicom image pixels
            img_1 = dcmH1.pixel_array.astype('uint16')
            img_2 = dcmH2.pixel_array.astype('uint16')
            img_3 = dcmH3.pixel_array.astype('uint16')
            
            try:
                laterality = dcmH1.ImageLaterality
            except:
                laterality = dcmH1.Laterality
                
            try:
                view = dcmH1.ViewPosition
            except:
                view = 'CC'
                dcmH1.ViewPosition = view
            
            if applyFlatFielding:
                if view != 'CC':
                    calibration_map = np.flipud(calibration_map)
                elif laterality != 'L':
                    calibration_map = np.fliplr(calibration_map)
                    
                if applyMeanFactor:
                    img_1[step:-step,:] = img_1[step:-step,:]/calibration_map[i,:,:]
                    img_1 = (img_1/mean_factor[i]).astype('uint16')
                    img_2[step:-step,:] = img_2[step:-step,:]/calibration_map[i+1,:,:]
                    img_2 = (img_2/mean_factor[i+1]).astype('uint16')
                    img_3[step:-step,:] = img_3[step:-step,:]/calibration_map[i+2,:,:]
                    img_3 = (img_3/mean_factor[i+2]).astype('uint16')
                else:
                    img_1[step:-step,:] = (img_1[step:-step,:]/calibration_map[i,:,:]).astype('uint16')
                    img_2[step:-step,:] = (img_2[step:-step,:]/calibration_map[i+1,:,:]).astype('uint16')
                    img_3[step:-step,:] = (img_3[step:-step,:]/calibration_map[i+2,:,:]).astype('uint16')
                
            # img_1 = log_scale(img_1,maxV).astype('uint16')
            # img_2 = log_scale(img_2,maxV).astype('uint16')
            
            h, w = img_1.shape
            if h < cropped_dim[0] or w < cropped_dim[1]:
                skip_crop_flag = True
                
            if crop_flag == True and skip_crop_flag == False:
                # The crop flag should be used if metrics are intended to be calculated so as to reduce memory usage
                img_1, img_2, img_3 = crop_imgs(img_1, img_2, img_3, cropped_dim, laterality, view, step, machine)

            #      # Downsample by 2
            # img_1 = img_1[::2,::2]
            # img_2 = img_2[::2,::2]
                
            # Forward through model
            interp_img_1, interp_img_2 = model_forward(model = model, img_1 = img_1, img_3 = img_3,
                                       maxV = maxV,                                  
                                       normalization = normalization,
                                       angle = angles[2*i+1],
                                       useTuningP = useTuningP)
            
            if correct_intensity_flag:
                # Correct intensity of interpolated image based on the GT
                interp_img_1 = correct_intensity(interp_img_1,img_1,img_3, maxV)
                interp_img_2 = correct_intensity(interp_img_2,img_1,img_3, maxV)
            
            # img_1 = exp_scale(img_1,maxV)
            # interp_img = exp_scale(interp_img,maxV)
            
            # Copy the restored data to the original dicom header
            dcmH1.PixelData = img_1.astype('uint16').copy('C')
            dcmH2.PixelData = img_2.astype('uint16').copy('C')
            dcmH3.PixelData = img_3.astype('uint16').copy('C')
            
            if crop_flag == True and skip_crop_flag == False:
                
                # Rewriting in the header files        
                dcmH1.Rows, dcmH1.Columns = cropped_dim                
                dcmH2.Rows, dcmH2.Columns = cropped_dim               
                dcmH3.Rows, dcmH3.Columns = cropped_dim
   
            # Write dicom files

            dcmH1.save_as(os.path.join(folder_name,"_{:0>2}.dcm".format(str(2*i))))
            # print("Saving {:0>2} as the first real image in sequence".format(str(2*i)))
              
            dcmH1.PixelData = interp_img_1.astype('uint16').copy(order='C')  
            dcmH1.save_as(os.path.join(folder_name,"_{:0>2}.dcm".format(str(2*i+1))))
            # print("Saving {:0>2} as the first interpolated image".format(str(2*i+1)))
        
            dcmH2.save_as(os.path.join(folder_name,"_{:0>2}.dcm".format(str(2*i+2))))
            # print("Saving {:0>2} as the middle real image in sequence".format(str(2*i+2)))    
            
            dcmH2.PixelData = interp_img_2.astype('uint16').copy(order='C')  
            dcmH2.save_as(os.path.join(folder_name,"_{:0>2}.dcm".format(str(2*i+3))))
            # print("Saving {:0>2} as the second interpolated image".format(str(2*i+3)))
            
            dcmH3.save_as(os.path.join(folder_name,"_{:0>2}.dcm".format(str(2*i+4))))
        
    return

#%%

if __name__ == '__main__':
    
    # ap = argparse.ArgumentParser(description='Generate interpolated DBT projections')
    # ap.add_argument("--mod", type=str, required=True, 
    #                 help="Image set from: (train_VCT, 'train_ClinicalHologic')")
    # ap.add_argument("--dts", type=int, required=True, 
    #                 help="Dataset size")
    # ap.add_argument("--loss", type=str, required=True, 
    #                 help="Loss type from: (Charb, PL4))
    # ap.add_argument("--norm", type=str, required=True, 
    #                 help="scale: normalize with vmin and vmax; zscore: standardize with mean and std")    
    # # ap.add_argument("--crop", type=bool, required=True,
    # #                 help="Crop images (True(1)/False(0))")

    
    # args = vars(ap.parse_args())
    
    # partition = args['mod']
    # dts = args['dts']
    # loss = args['loss']
    # normalization =  args['norm']
    # crop = args['crop']           
    
    # Args declaration for debugging        

    machine = 'Hologic'
    partition = 'train_ClinicalHologic'
    dts = 9906
    loss = 'PL4'
    # loss = 'Charb'
    normalization = 'scale' #'zscore' # or 'scale'
    crop = True
    map_intensity = True
    applyFlatFielding = False
    
    cropped_dim = (1536, 608)
    new_dim = (2048,1152)
   
    useTuningP = False
    num_proj = 15
    angular_range = 15    
    
    maxV = 1023.
    # maxV = 4095.
    # prop = np.load('tools/{}_dataset_prop_{}.npy'.format(partition,applyFlatFielding*'wFlatFieldingCorrection'),allow_pickle=True).item()
    # maxV = prop['max']

    path_data = '/media/laviusp/c2370571-c46d-4175-acba-c89fc1b3e499/lavi/Documents/Arthur/Inrad_Processed//'
    # subset defines the folder to read the exams from. Projections will be generated for those DBTs
    subset = 'test'
    path_models = "final_models/mod_{}/{}_{}{}/".format(partition,dts,typ,useTuningP*"_wAngle")
    path2write = "outputs"
    
    makedir(path2write)
    
    path_final_generator = path_models + "model_DBT_PVInterpolation-{}Triplets_{}_NormType-{}{}_3ch_{}deg{}projs{}.pth".format(cropped_dim,
                                                                                                                                machine,
                                                                                                                                normalization,
                                                                                                                                applyFlatFielding*'_wFlatFieldingCorrection',
                                                                                                                                angular_range,
                                                                                                                                num_proj,
                                                                                                                                useTuningP*'_wAngle')
    
    # To use specific address of model trained in previous versions
    # path_final_generator = 'final_models/mod_HologicClinical/Charb+PL/model_RRIN_DBT_VCT_raw_L=8191_Factor1_(1792, 608)Slice_charbinit_perceptual_vgg_Epoch_3'
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    if useTuningP:
        generator = Gen()
    else:
        generator = Net_t(input_sz=cropped_dim)
    
    # Send it to device (GPU if exist)
    generator = generator.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    generator, epoch = load_model(generator,path_final_model=path_final_generator,oldModel = False)
    # epoch = 4
    
    print("Running test on {}.".format(device))
    
    test(model = generator,
         path_data = path_data, 
         path2write = path2write,
         partition = partition,
         subset = subset,
         machine = machine,
         crop_flag = crop,
         correct_intensity_flag = map_intensity,
         cropped_dim = new_dim, 
         loss = loss, 
         epoch = epoch, 
         nPv = num_proj,
         angR = angular_range,
         maxV = maxV,
         normalization = normalization,
         applyFlatFielding = applyFlatFielding)


