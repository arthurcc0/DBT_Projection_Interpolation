#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:55:44 2023

@author: ArthurC
"""

import numpy as np
import os
import pydicom as dicom
from pathlib import Path
from skimage.filters import threshold_otsu

#%%

if __name__ == '__main__':
        
    
    path_data = '/media/laviusp/c2370571-c46d-4175-acba-c89fc1b3e499/lavi/Documents/Arthur/Lund_Images/'
    # partition = 'clinical_DBT'
    partition = 'train_VCT'
    
    # Flag to apply or not the mean factor to balance the mean value along the projections
    applyMeanFactor = False
    applyFlatFielding = False
    step = 250
    calibration_map = np.load(os.path.join(path_data,'SiemensProjCalibrationMap_{}_step_{}.npy'.format('L',step)))
    mean_factor = np.load(os.path.join(path_data,'SiemensProjMeanFactor.npy'))
    
    folder_names = [str(item) for item in Path(path_data).joinpath(partition).glob("*") if Path(item).is_dir()]
    
    img_min_set = 2**12
    img_max_set = 0
    img_means = []
    img_vars = []
    
    folder_names = [str(item) for item in Path(path_data).joinpath(partition).glob("*") if Path(item).is_dir()] #.glob if any specific folder is prefered

    types = ['*.dcm','*.DCM','*.IMA']   
    
    for folder_name in folder_names:
        
        paths = []
        for t in types:
            paths += Path(folder_name).glob(t)
        all_projections_inside = sorted([str(item) for item in paths],key=lambda k: (len(k), k))
    
        # Loop on each projection
        for i,proj in enumerate(all_projections_inside):
            
            img_hdr = dicom.read_file(proj)
            img = img_hdr.pixel_array[step:-step,:]
            
            if applyFlatFielding:
                laterality = img_hdr.ImageLaterality
                view = img_hdr.ViewPosition
                if view != 'CC':
                    calibration_map = np.flipud(calibration_map)
                elif laterality != 'L':
                    calibration_map = np.fliplr(calibration_map)
                    
                if applyMeanFactor:
                    img = img/calibration_map[i,:,:]
                    img = (img/mean_factor[i]).astype('uint16')
                else:
                    img = (img/calibration_map[i,:,:]).astype('uint16')

            mask = img < threshold_otsu(img) 

            img_min = img[mask].min()
            img_max = img[mask].max()
            
            if(img_min < img_min_set):
                img_min_set = img_min
            if(img_max > img_max_set):
                img_max_set = img_max
                
            img_means.append(np.mean(img[mask]))
    
    img_mean_set = np.mean(img_means)
    
    for folder_name in folder_names:
        
        paths = []
        for t in types:
            paths += Path(folder_name).glob(t)
        all_projections_inside = sorted([str(item) for item in paths],key=lambda k: (len(k), k))
    
        # Loop on each projection
        for i,proj in enumerate(all_projections_inside):
            
            img_hdr = dicom.read_file(proj)
            img = img_hdr.pixel_array[step:-step,:]
            
            if applyFlatFielding:
                laterality = img_hdr.ImageLaterality
                view = img_hdr.ViewPosition
                if view != 'CC':
                    calibration_map = np.flipud(calibration_map)
                elif laterality != 'L':
                    calibration_map = np.fliplr(calibration_map)
                    
                if applyMeanFactor:
                    img = img/calibration_map[i,:,:]
                    img = (img/mean_factor[i]).astype('uint16')
                else:
                    img = (img/calibration_map[i,:,:]).astype('uint16')
                    
            mask = img < threshold_otsu(img)
            img_vars.append(np.mean((img[mask] - img_mean_set) ** 2))
            
    img_std_set = np.sqrt(np.mean(img_vars,axis = 0))
    
    print("Inside breast area: Min {} , Max {}, Mean {} and std {}".format(img_min_set,img_max_set,img_mean_set,img_std_set))
    
    prop = {'min':img_min_set,'max':img_max_set,'mean':img_mean_set,'std':img_std_set}
    np.save('{}_dataset_prop_{}.npy'.format(partition,applyFlatFielding*'wFlatFieldingCorrection'),prop)
