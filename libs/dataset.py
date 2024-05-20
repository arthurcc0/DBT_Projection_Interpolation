"""
Created on Fri Sep 17 10:10:37 2021

@author: Arthur C.
"""

import torch
import h5py
from numpy import log,exp

def de_scale(data, vmax):
    
    data = data * vmax 
        
    return data

def scale(data, vmin, vmax):
    
    # data -= vmin
    # data /= red_factor
    # data += vmin
    data /= vmax
    
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
          
    return data

def zscore(data, mean, std):
    
    data = data - mean
    
    return data/std

def de_zscore(data, mean, std):
    
    data = data * std
    
    return (data + mean).astype('uint16')

def log_scale(data, vmax):
    
    c = vmax/log(1+vmax)
    
    return c*log(data+1)

def exp_scale(data, vmax):
    
    c = vmax/log(1+vmax)
    
    return (exp(data/c)-1)

class DBTTripletDataset(torch.utils.data.Dataset):
  """ DBT Projection View's Triplets dataset."""
  def __init__(self, h5_file_name, normalization, log = False, vmin = 0., vmax = 4095., mean = 240., std = 100.):
    """
    Args:
      h5_file_name (string): Path to the h5 file.
    """
    self.h5_file_name = h5_file_name
    self.vmin = vmin
    self.vmax = vmax
    self.mean = mean
    self.std = std

    self.h5_file = h5py.File(self.h5_file_name, 'r')

    self.img_1 = self.h5_file['img_1']
    self.target = self.h5_file['target']
    self.img_3 = self.h5_file['img_3']
    self.target_angle= self.h5_file['target_angles']
    
    self.normalization = normalization
    self.log = log # If log is True, z-score can't be applied. Mean and std have to be recalculated

  def __len__(self):
      
    return self.img_1.shape[0]

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_1 = self.img_1[idx,:,:].astype('float')
    target = self.target[idx,:,:].astype('float')
    img_3 = self.img_3[idx,:,:].astype('float')
    target_angle = self.target_angle[idx].astype('float')
    
    if self.log:  
        img_1 = log_scale(img_1, self.vmax)
        target = log_scale(target, self.vmax)
        img_3 = log_scale(img_3, self.vmax)
    
    # Normalize 0-1 data
    if self.normalization == 'scale':
        img_1 = scale(img_1, self.vmin, self.vmax)
        target = scale(target, self.vmin, self.vmax)
        img_3 = scale(img_3, self.vmin, self.vmax)
    # Z-score: standardize date according to mean value of dataset and standard deviation
    elif self.normalization == 'zscore':
        img_1 = zscore(img_1, self.mean, self.std)
        target = zscore(target, self.mean, self.std)
        img_3 = zscore(img_3, self.mean, self.std)
    
    # To torch tensor
    img_1 = torch.from_numpy(img_1.astype(float)).type(torch.FloatTensor)
    target = torch.from_numpy(target.astype(float)).type(torch.FloatTensor)
    img_3 = torch.from_numpy(img_3.astype(float)).type(torch.FloatTensor)
    target_angle = torch.Tensor([target_angle]).type(torch.FloatTensor)
    
    return img_1, target, img_3, target_angle

