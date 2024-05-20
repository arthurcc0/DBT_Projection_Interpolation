"""
Created on Fri Sep 17 10:10:37 2021

@author: Rodrigo
"""

import torch
import sys

from tqdm import tqdm 

sys.path.insert(0, '../')

from libs.dataset import DBTTripletDataset


if __name__ == '__main__':
        
    
    path_data = '/home/laviusp/Documents/Arthur/'
    partition = 'train'
    
    cropped_dim = (1664, 608)
    nTriplets_total = 9906
    
    dataset_path = '{}DBT_ProjInterpolation_{}_{}Triplets_Hologic_{}.h5'.format(path_data, partition, cropped_dim, nTriplets_total)
    
    img_min = 2**12
    img_max = 0
    target_min = 2**12
    target_max = 0
    
    # Create dataset helper
    train_set = DBTTripletDataset(dataset_path,vmin=target_max,vmax=target_min, normalization=False)
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=1, 
                                              shuffle=True,
                                              pin_memory=True)
    
    for step, (img_1, target, img_3) in enumerate(tqdm(train_loader)):
        
        img_1_min_batch = img_1.min()
        img_1_max_batch = img_1.max()
        
        target_min_batch = target.min()
        target_max_batch = target.max()
        
        img_3_min_batch = img_3.min()
        img_3_max_batch = img_3.max()
        
        if(img_1_min_batch < img_min):
            img_min = img_1_min_batch
        if(img_1_max_batch > img_max):
            img_max = img_1_max_batch
            
        if(target_min_batch < target_min):
            target_min = target_min_batch
        if(target_max_batch > target_max):
            target_max = target_max_batch
        
        if(img_3_min_batch < img_min):
            img_min = img_3_min_batch
        if(img_3_max_batch > img_max):
            img_max = img_3_max_batch
            
    print(img_min, img_max) 
    print(target_min, target_max) 