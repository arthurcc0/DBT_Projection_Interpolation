#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:17:38 2024

@author: Arthur

Load the projection sequences of a folder of exams and save as triplets in a h5 file
Modified version adding the target angle information
"""

import numpy as np
import pydicom as dicom
import h5py
import random
from pathlib import Path
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

#%%

def process_each_folder(folder_name, step, num_proj=25, angular_range = 50,  cropped_dim = (1664, 608)):
    '''Process DBT folder to extract triplets'''
    
    types = ['*.dcm','*.DCM','*.IMA']   
    paths = []
    for t in types:
        paths += Path(folder_name).glob(t)
    all_projections_inside = sorted([str(item) for item in paths],key=lambda k: (len(k), k))
    
    global nt_imgs
    
    triplets = []
    target_angles = []
    
    # Angles corresponding to the acquisition geometry and equipment
    angles = np.linspace(-angular_range/2,angular_range/2,num_proj)
    
    # Loop on each projection
    for i,proj in enumerate(all_projections_inside[:num_proj-2]):
        
        # First image of sequence
        img_1_header = dicom.read_file(proj)
        img_1 = img_1_header.pixel_array
        # Second image of sequence (reference)
        img_2 = dicom.read_file(all_projections_inside[i+1]).pixel_array
        # Third image of sequence
        img_3 = dicom.read_file(all_projections_inside[i+2]).pixel_array
        
        # Corresponding angle of the target angle
        target_angles.append(angles[i+1])
        
        assert img_1.shape == img_2.shape, "image sizes differ"
    
        global throw_away
        
        try:
            laterality = img_1_header.ImageLaterality
        except:
            laterality = img_1_header.Laterality
            
        try:
            view = img_1_header.ViewPosition
        except:
            view = 'CC'
            img_1_header.ViewPosition = view
        
        
        # Get image shape
        h, w = img_2.shape
        
        # Margin to centralize cropped region on the middle of breast if view = CC
        margin = (h - cropped_dim[0])//2
    
        # Crop according to view and laterality
        # ATTENTION - Make sure that the images are not all fliped to the right side, different equipment has different protocol
        # Hologic: Images are always on the right side. Siemens: Left breasts are displayed at the left side
        if view == 'CC':
            img_1 = img_1[margin:cropped_dim[0]+margin,-cropped_dim[1]:]
            img_2 = img_2[margin:cropped_dim[0]+margin,-cropped_dim[1]:]
            img_3 = img_3[margin:cropped_dim[0]+margin,-cropped_dim[1]:]
        elif laterality == 'R':
            img_1 = img_1[:cropped_dim[0],-cropped_dim[1]:]
            img_2 = img_2[:cropped_dim[0],-cropped_dim[1]:]
            img_3 = img_3[:cropped_dim[0],-cropped_dim[1]:]
        else:
            img_1 = img_1[-cropped_dim[0]:,-cropped_dim[1]:]
            img_2 = img_2[-cropped_dim[0]:,-cropped_dim[1]:]
            img_3 = img_3[-cropped_dim[0]:,-cropped_dim[1]:] 
    
        if not img_1.any():
            print("Folder: {} Laterality: {} View: {}".format(folder_name,laterality,view))
            break;
        
        triplets.append((img_1, img_2, img_3))
                    
    return triplets, target_angles

#%%

if __name__ == '__main__':
    
    path2read = '/' # Path where to retrieve the input images
    path2write = '/' # Path to save the .h5 file
    
    # Flag to display images while creating the h5 file
    displayImgs = True
    
    # Flag to apply or not the mean factor to balance the mean value along the projections
    applyMeanFactor = False

    step = 0 # Step value to disconsider part of the edges of the projections to perform cropping of intensity correction
    
    partition = 'train'
    folder_names = [str(item) for item in Path(path2read).joinpath(partition).glob("*") if Path(item).is_dir()] #.glob if any specific folder is prefered
    
    random.shuffle(folder_names)
    
    np.random.seed(0)
    
    machine = 'Hologic'
    # machine = 'Siemens'
    
    # Cropped region to fit in network input layer (set it according to memory available)
    cropped_dim = (1536, 608)
    # cropped_dim = (1792,640)
    
    # Number of projections views in DBT image sets
    num_proj = 15
    
    angular_range = 15
    
    nTriplets_total = (num_proj-2)*(len(folder_names)+1)
    
    throw_away = 0
    flag_final = 0
    
    nTriplets = 0
    # Create h5 file
    f = h5py.File('{}DBT_ProjInterpolation_{}_{}Triplets_{}_{}_{}deg{}projs.h5'.format(path2write, partition, cropped_dim,machine, nTriplets_total,angular_range,num_proj), 'a')
    
    # Loop on each DBT folder (projections)
    for idX, folder_name in enumerate(folder_names):
        
        # Get all possible triplets from one exam
        triplets, target_angles = process_each_folder(folder_name = folder_name, num_proj=num_proj, cropped_dim = cropped_dim,
                                       angular_range = angular_range,
                                       # applyFlatFielding = applyFlatFielding, applyMeanFactor = applyMeanFactor,
                                       # calibration_map = calibration_map, mean_factor = mean_factor,
                                       step = step)        
                
        imgs_1 = np.stack([x[0] for x in triplets])
        target = np.stack([x[1] for x in triplets])
        imgs_3 = np.stack([x[2] for x in triplets])
        
        if idX%1000 == 0 and displayImgs == True: 
            figure = plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(imgs_1[0],'gray')
            plt.title("First image in sequence"); plt.grid(False)
            
            plt.subplot(1,3,2)
            plt.imshow(target[0],'gray')
            plt.title("Second image in sequence (target)"); plt.grid(False)
            
            plt.subplot(1,3,3)
            plt.imshow(imgs_3[0],'gray')
            plt.title("Third image in sequence"); plt.grid(False)
        
        imgs_1 = np.expand_dims(imgs_1, axis=1) 
        target = np.expand_dims(target, axis=1)
        imgs_3 = np.expand_dims(imgs_3, axis=1) 
        
        nTriplets += imgs_1.shape[0]
        
        # Did I reach the expected size (nTriplets_total)?
        if  nTriplets >= nTriplets_total:
            flag_final = 1
            # diff = nTriplets_total - nTriplets
            # imgs_1 = imgs_1[:diff,:,:,:]
            # target = target[:diff,:,:,:]
            # imgs_3 = imgs_3[:diff,:,:,:]
                            
        if idX == 0:
            f.create_dataset('img_1', data=imgs_1, chunks=True, maxshape=(None,1,cropped_dim[0],cropped_dim[1]))
            f.create_dataset('target', data=target, chunks=True, maxshape=(None,1,cropped_dim[0],cropped_dim[1])) 
            f.create_dataset('img_3', data=imgs_3, chunks=True, maxshape=(None,1,cropped_dim[0],cropped_dim[1]))
            f.create_dataset('target_angles', data=target_angles, chunks=True, maxshape=(None,)) 
        else:
            f['img_1'].resize((f['img_1'].shape[0] + imgs_1.shape[0]), axis=0)
            f['img_1'][-imgs_1.shape[0]:] = imgs_1
            
            f['target'].resize((f['target'].shape[0] + target.shape[0]), axis=0)
            f['target'][-target.shape[0]:] = target
            
            f['img_3'].resize((f['img_3'].shape[0] + imgs_3.shape[0]), axis=0)
            f['img_3'][-imgs_3.shape[0]:] = imgs_3
            
            f['target_angles'].resize((f['target_angles'].shape[0] + np.shape(target_angles)[0]), axis=0)
            f['target_angles'][-np.shape(target_angles)[0]:] = target_angles
            
        print("Iter {} and 'img_1' chunk has shape:{}, 'target':{}, 'img_3':{} and 'target_angles':{}".format(idX,f['img_1'].shape,f['target'].shape,f['img_3'].shape,f['target_angles'].shape))
        
        if flag_final:
            break

    f.close()       
     
    
    
