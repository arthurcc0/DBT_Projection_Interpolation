# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 11:35:17 2022

@author: Arthur

rescale and save original projections as .dcm
"""

import os
import pydicom
#%%
dim = (2048,1152)
num_projs = 15

path_data = '/media/laviusp/c2370571-c46d-4175-acba-c89fc1b3e499/lavi/Documents/Arthur/Inrad_Processed/'
subset = 'test'

out_path = os.path.join(path_data,subset+'_cropped')

#%% Save as DICOM - Generate even projections of a set of DBT projections
all_exams = sorted(os.listdir(os.path.join(path_data,subset)),key=lambda k: (len(k), k))
for exam in all_exams:
    
    exam_folder =  os.path.join(path_data,subset,exam)
    
    new_folder = os.path.join(out_path,exam)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        
    all_projections_inside = sorted(os.listdir(exam_folder),key=lambda k: (len(k), k))
    sample = pydicom.dcmread(os.path.join(exam_folder,all_projections_inside[0]))
    try:
        view = [sample.ImageLaterality,sample.ViewPosition]
    except(RuntimeError,AttributeError):
        view = ['R','CC']
        
    h,w = sample.pixel_array.shape[:2]
    margin = (h-dim[0])//2
    
    # Loop on each projection
    for i,proj in enumerate(all_projections_inside):
        
        # First image of sequence
        img_header = pydicom.dcmread(os.path.join(exam_folder,proj))
        img = img_header.pixel_array

        if view[-1] == 'CC':
            if view[0] == 'R':
                img = img[margin:dim[0]+margin,-dim[1]:]
            else: 
                img = img[margin:dim[0]+margin,:dim[1]]
        elif view[0] == 'R':
            img = img[:dim[0],-dim[1]:]
        else:
            img = img[:dim[0],:dim[1]]

        img_header.PixelData = img.copy(order='C')   
        img_header.Rows, img_header.Columns = dim
        pydicom.dcmwrite("{}/{:02d}.dcm".format(new_folder,i),
                         img_header)