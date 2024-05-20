"""
Created on Fri Sep 17 10:10:37 2021

@author: ArthurC
"""

import matplotlib.pyplot as plt
import torch
import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Own codes
from libs.models import Gen,Net_t, Vgg16
from libs.utilities import load_model, image_grid, makedir
from libs.dataset import DBTTripletDataset

import libs.pytorch_ssim

#%%

def train(generator, vgg, gOptimizer, epoch, train_loader, device, summarywriter):
    
    # Enable trainning
    generator.train()

    for step, (img_1, target, img_3, target_angle) in enumerate(tqdm(train_loader)):

        img_1 = img_1.to(device)
        target = target.to(device)
        img_3 = img_3.to(device)
        target_angle = target_angle.to(device)
        
        # Zero all grads            
        gOptimizer.zero_grad()
            
        # Generate a batch of new images
        gen_data = generator(img_1,img_3,pos = target_angle)    # t=0.5   
        
        # PL
        # features_y = vgg(gen_data)
        # features_x = vgg(target)
        
        # pl4_loss = torch.mean((features_y.relu4_3 - features_x.relu4_3)**2)
        pl4_loss = vgg(target,gen_data)
        loss = pl4_loss 
        
        ### Backpropagation ###
        # Calculate all grads
        loss.backward()
        
        # Update weights and biases based on the calc grads 
        gOptimizer.step()
        
        # ---------------------
    
        # Write Gen Loss to tensorboard
        summarywriter.add_scalar('Model_Loss/train', 
                                 loss.item(), 
                                 epoch * len(train_loader) + step)
        
        
        # Print images to tensorboard img_1, target_img, img_3, interp_img
        if step % 1000 == 0:
            image_grid(img_1[0,0,:,:], 
                       target[0,0,:,:], 
                       img_3[0,0,:,:],
                       gen_data[0,0,:,:])
            summarywriter.add_figure('Plot/train', 
                                     image_grid(img_1[0,0,:,:], 
                                                target[0,0,:,:], 
                                                img_3[0,0,:,:],
                                                gen_data[0,0,:,:]),
                                     epoch * len(train_loader) + step,
                                     close=True)
            # Write Gen SSIM to tensorboard
            summarywriter.add_scalar('Gen_SSIM/train', 
                                     ssim(gen_data, target).item(), 
                                     epoch * len(train_loader) + step)
        
        
#%%

if __name__ == '__main__':
    
    # ap = argparse.ArgumentParser(description='Train a model to interpolate between DBT projection views with Perceptual Loss (PL4)')
    # ap.add_argument("--mod", type=str, required=True, 
    #                 help="Image set from: (train_VCT, 'train_ClinicalHologic')")
    # ap.add_argument("--dts", type=int, required=True, 
    #                 help="Dataset size")
    # ap.add_argument("--norm", type=str, required=True, 
    #                 help="scale: normalize with vmin and vmax; zscore: standardize with mean and std")  
    
    # args = vars(ap.parse_args())
    
    # partition = args['mod']
    # dts = args['dts']
    # normalization =  args['norm']
    
     
    # Args declaration for debugging
    # partition = 'clinical_DBT'
    partition = 'train_ClinicalHologic'
    dts = 9906
    normalization = 'scale'
    
    cropped_dim = (1536, 608) # Size of the image was set based on GPÙ memory
    applyFlatFielding = False
    machine = 'Hologic'
    # machine = 'Siemens'
    
    num_proj = 15
    angular_range = 15
    
    # flag to set the use of the tuning parameter incorporated in the network
    useTuningP = False
    
    path_data = '/home/laviusp/Documents/Arthur/'
    path_models = "final_models/mod_{}/{}_PL4{}/".format(partition,dts,useTuningP*"_wAngle")
    path_logs = "final_logs/mod_{}/{}/{}".format(partition,dts,time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))
    
    
    path_final_generator = path_models + "model_DBT_PVInterpolation-{}Triplets_{}_NormType-{}{}_3ch_{}deg{}projs{}.pth".format(cropped_dim,
                                                                                                                                machine,
                                                                                                                                normalization,
                                                                                                                                applyFlatFielding*'_wFlatFieldingCorrection',
                                                                                                                                angular_range,
                                                                                                                                num_proj,
                                                                                                                                useTuningP*'_wAngle')
    path_final_critic = path_models + "critic_DBT_PVInterpolation-{}Triplets_{}_NormType-{}{}_3ch_{}deg{}projs{}.pth".format(cropped_dim,
                                                                                                                              machine,
                                                                                                                              normalization,
                                                                                                                              applyFlatFielding*'_wFlatFieldingCorrection',
                                                                                                                              angular_range,
                                                                                                                              num_proj,
                                                                                                                              useTuningP*'_wAngle')
    
    LR = 1e-5
    batch_size = 1
    n_epochs = 5
    
    dataset_path = '{}DBT_ProjInterpolation_{}_{}Triplets_{}_{}_{}deg{}projs.h5'.format(path_data,
                                                                                 partition,
                                                                                 cropped_dim,
                                                                                 machine,
                                                                                 dts,
                                                                                 angular_range,
                                                                                 num_proj)
    
    # Tensorboard writer
    summarywriter = SummaryWriter(log_dir=path_logs)
    
    makedir(path_models)
    makedir(path_logs)
    
    # Test if there is a GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    if useTuningP:
        generator = Gen()
    else:
        generator = Net_t(level=4,input_sz=cropped_dim)    
    
    # Create the optimizer and the LR scheduler
    optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5)    
    
    # Send it to device (GPU if exist)
    generator = generator.to(device)
    # critic = critic.to(device)
    
    # Load gen pre-trained model parameters (if exist)
    generator,start_epoch = load_model(generator, 
                              optimizer, 
                              scheduler,
                                path_final_model=path_final_generator, # Resume training
                                # path_pretrained_model= "final_models/mod_{}/{}_Charb/model_DBT_PVInterpolation-{}Triplets_Siemens_NormType-{}_{}.pth".format(partition,dts,cropped_dim,normalization,applyFlatFielding*'wFlatFieldingCorrection') # Start from a pretrained
                                # path_pretrained_model = "final_models/mod_{}/{}_PL4/model_DBT_PVInterpolation-{}Triplets_Siemens_NormType-{}_{}.pth".format('SiemensVCT',2323,cropped_dim,normalization,applyFlatFielding*'wFlatFieldingCorrection') # Start from a pretrained
                                )

    # Create dataset helper

    # Load proprieties of the dataset (min, max, mean and std)
    # prop = load('tools/{}_dataset_prop_{}.npy'.format(partition,applyFlatFielding*'wFlatFieldingCorrection'), allow_pickle=True).item()
    # Choose to normalize following min and max or to use Z-score with mean and std
    train_set = DBTTripletDataset(dataset_path,
                                  normalization = normalization, # 'scale' (normalization using vmin and vmax) or 'zscore' (standardization using mean and std)
                                  log = False,
                                   # vmin = prop['min'],
                                    # vmax = prop['max'],
                                  vmin = 0.,
                                   vmax = 1023.,
                                  # vmax = 4095., 
                                  # mean = prop['mean'],
                                  # std = prop['std']
                                  )
    
    # Create dataset loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              pin_memory=True)
    
    ssim = libs.pytorch_ssim.SSIM(window_size = 11)

    vgg = Vgg16(requires_grad=False).to(device)
        
    # Loop on epochs
    for epoch in range(start_epoch, n_epochs + start_epoch):
        
      print("Epoch:[{}] LR:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    
      # Train the model for 1 epoch
      train(generator,
             vgg,
             optimizer, 
             epoch, 
             train_loader, 
             device, 
             summarywriter) 
    
      # Update LR
      scheduler.step()
    
      # Save the model
      torch.save({
                 'epoch': epoch,
                 'model_state_dict': generator.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 }, path_final_generator)
      
      # if (epoch + 1) % 5 == 0:
          # Testing code
          # os.system("python main_testing.py --mod {} --dts {} --loss PL4 --norm {} --crop True".format(partition, dts, normalization))

                       