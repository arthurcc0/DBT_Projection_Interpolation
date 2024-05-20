"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo and adapted by ArthurC
"""

import os 
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pydicom
from collections import OrderedDict

def load_model(model, optimizer=None, scheduler=None, path_final_model='', path_pretrained_model='',oldModel = False):
    """Load pre-trained model, resume training or initialize from scratch."""
    
    epoch = 0
      
    # Resume training
    if os.path.isfile(path_final_model):
          
      checkpoint = torch.load(path_final_model)
      # Adapt old model dictionary keys to be readable
      if oldModel:
          cp = OrderedDict([('generator.'+k,v) for k,v in checkpoint['model_state_dict'].items()])
          checkpoint['model_state_dict'] = cp
          checkpoint['epoch'] = checkpoint['prev_epoch']
      
      model.load_state_dict(checkpoint['model_state_dict'])
      if optimizer != None:
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if scheduler != None:
          scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      epoch = checkpoint['epoch'] + 1
      
      print('Loading model {} from epoch {}.'.format(path_final_model, epoch-1))
      
    # Loading pre-trained model
    elif os.path.isfile(path_pretrained_model):
          
      # Load a pre trained network 
      checkpoint = torch.load(path_pretrained_model)
      model.load_state_dict(checkpoint['model_state_dict'])
      
      print('Initializing from scratch \nLoading pre-trained {}.'.format(path_pretrained_model))
      
    # Initializing from scratch
    else:
      print('I couldnt find any model, I am just initializing from scratch.')
      
    return model,epoch


def image_grid(img_1, target_img, img_3, interp_img):
    """Return a 1x3 grid of the images as a matplotlib figure."""
    
    # Get from GPU
    img_1 = img_1.to('cpu')
    target_img = target_img.to('cpu')
    interp_img = interp_img.to('cpu').detach()
    img_3 = img_3.to('cpu')
    
    # Create a figure to contain the plot.
    figure = plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(torch.squeeze(img_1),'gray')
    plt.title("First image in sequence"); plt.grid(False)
    
    plt.subplot(1,4,2)
    plt.imshow(torch.squeeze(target_img),'gray')
    plt.title("Second image in sequence (ref.)"); plt.grid(False)
    
    plt.subplot(1,4,3)
    plt.imshow(torch.squeeze(img_3),'gray')
    plt.title("Third image in sequence"); plt.grid(False)
    
    plt.subplot(1,4,4)
    plt.imshow(torch.squeeze(interp_img),'gray')
    plt.title("Interpolated image (second in sequence)"); plt.grid(False)
    
    return figure

def makedir(path2create):
    """Create directory if it does not exists."""
 
    error = 1
    
    if not os.path.exists(path2create):
        os.makedirs(path2create)
        error = 0
    
    return error

def readDicom(dir2Read, imgSize = (3584,2816)):
    """Read Dicom function."""
      
    # List dicom files
    dcmFiles = sorted([p for p in os.listdir(dir2Read) if p.endswith(".dcm") or p.endswith(".IMA")],key=lambda k: (len(k), k))
    
    if not dcmFiles:    
        raise ValueError('No DICOM files found in the specified path.')
    
    dcmImg = np.empty([imgSize[0],imgSize[1],len(dcmFiles)])
    dcmH = pydicom.dcmread(os.path.join(dir2Read,str(dcmFiles[0])))
    
    # Get image shape
    h, w = dcmH.Rows, dcmH.Columns
    
    if (h != imgSize[0] or w != imgSize[1]):
        
        crop_flag = True
        
        laterality = dcmH.ImageLaterality
        try:
            view = dcmH.ViewPosition
        except:
            view = 'CC'
        
        # Margin to centralize cropped region on the middle of breast if view = CC
        margin = (h - imgSize[0])//2   

    if crop_flag:
        for ind,dcm in enumerate(dcmFiles):
        
            dcmH = pydicom.dcmread(os.path.join(dir2Read,str(dcm)))
            
            # Crop according to view and laterality
            if view == 'CC':
                if laterality == 'R':
                    dcmImg[:,:,ind] = dcmH.pixel_array[margin:imgSize[0]+margin,-imgSize[1]:].astype('float32')
                else: 
                    dcmImg[:,:,ind] = dcmH.pixel_array[margin:imgSize[0]+margin,:imgSize[1]].astype('float32')
            elif laterality == 'R':
                dcmImg[:,:,ind] = dcmH.pixel_array[-imgSize[0]:,-imgSize[1]:].astype('float32')
            else:
               dcmImg[:,:,ind] = dcmH.pixel_array[-imgSize[0]:,:imgSize[1]].astype('float32')
         
    else:    
        for ind,dcm in enumerate(dcmFiles):
            
            dcmH = pydicom.dcmread(os.path.join(dir2Read,str(dcm)))
            
            dcmImg[:,:,ind] = dcmH.pixel_array.astype('float32')

    return dcmImg

