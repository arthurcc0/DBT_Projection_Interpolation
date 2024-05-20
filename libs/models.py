"""
Created on Fri Sep 17 10:10:37 2021

@author: ArthurC 

Adapted to use different UNet modules. Also contains the VGG16 network PL loss class

"""

import torch 
from torch import nn
from torchvision import models
from collections import namedtuple
import torch.nn.functional as F
import numpy as np
from libs.unet import UNet
from libs.tunet import tUNet
from matplotlib import pyplot as plt

def warp(img, flow):
    _, _, H, W = img.size()
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]
    x = gridX.unsqueeze(0).expand_as(u).float()+u
    y = gridY.unsqueeze(0).expand_as(v).float()+v
    normx = 2*(x/W-0.5)
    normy = 2*(y/H-0.5)
    grid = torch.stack((normx,normy), dim=3)
    warped = F.grid_sample(img, grid, align_corners=False) # Updated the align_corners parameter, but not sure if the original used it
    return warped

class Net(nn.Module):
    def __init__(self,level=4,wf=4,kernel_sz = 3):
        super(Net, self).__init__()      
        # Using the original UNets
        self.Mask = UNet(8,2,level,wf=wf,kernel_sz = kernel_sz) # Modified from: Unet(16,2,4) to fit 1ch images
        self.Flow_L = UNet(2,4,level+1,kernel_sz = kernel_sz) # Modified from: Unet(6,4,5) to fit 1ch images
        self.refine_flow = UNet(6,4,level,wf=wf,kernel_sz = kernel_sz) # Modified from: Unet(10,4,4) to fit 1ch images
        self.final = UNet(3,1,level,wf=wf,kernel_sz = kernel_sz) # Modified from: Unet(9,1,4) to fit 1ch images and to output 1ch

    def process(self,x0,x1,t=0.5):

        x = torch.cat((x0,x1),1)
        Flow = self.Flow_L(x)
        Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
        Flow_t_0 = -(1-t)*t*Flow_0_1+t*t*Flow_1_0
        Flow_t_1 = (1-t)*(1-t)*Flow_0_1-t*(1-t)*Flow_1_0
        Flow_t = torch.cat((Flow_t_0,Flow_t_1,x),1)
        Flow_t = self.refine_flow(Flow_t)
        Flow_t_0 = Flow_t_0+Flow_t[:,:2,:,:]
        Flow_t_1 = Flow_t_1+Flow_t[:,2:4,:,:]
        xt1 = warp(x0,Flow_t_0)
        xt2 = warp(x1,Flow_t_1)
        temp = torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
        Mask = torch.sigmoid(self.Mask(temp))
        w1, w2 = (1-t)*Mask[:,0:1,:,:], t*Mask[:,1:2,:,:]
        output = (w1*xt1+w2*xt2)/(w1+w2+1e-8)

        return output

    def forward(self, input0, input1, t=0.5):

        output = self.process(input0,input1,t)
        compose = torch.cat((input0, input1, output),1)
        final = self.final(compose)+output
        #final = final.clamp(0,1)

        return final
    
class Net_t(nn.Module):
    def __init__(self,input_sz=(1536,608),level=4,wf=4,kernel_sz = 3):
        super(Net_t, self).__init__()
        # Using the UNet with the tunable parameter
        self.Mask = tUNet(8,2,level,wf=wf,kernel_sz = kernel_sz,input_size=input_sz) # Modified from: Unet(16,2,4) to fit 1ch images
        self.Flow_L = tUNet(2,4,level+1,kernel_sz = kernel_sz,input_size=input_sz) # Modified from: Unet(6,4,5) to fit 1ch images
        self.refine_flow = tUNet(6,4,level,wf=wf,kernel_sz = kernel_sz,input_size=input_sz) # Modified from: Unet(10,4,4) to fit 1ch images
        self.final = tUNet(3,1,level,wf=wf,kernel_sz = kernel_sz,input_size=input_sz) # Modified from: Unet(9,1,4) to fit 1ch images and to output 1ch
        
    def process(self,x0,x1,pos,t=0.5):

        x = torch.cat((x0,x1),1)
        Flow = self.Flow_L(x,pos)
        Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
        Flow_t_0 = -(1-t)*t*Flow_0_1+t*t*Flow_1_0
        Flow_t_1 = (1-t)*(1-t)*Flow_0_1-t*(1-t)*Flow_1_0
        Flow_t = torch.cat((Flow_t_0,Flow_t_1,x),1)
        Flow_t = self.refine_flow(Flow_t,pos)
        Flow_t_0 = Flow_t_0+Flow_t[:,:2,:,:]
        Flow_t_1 = Flow_t_1+Flow_t[:,2:4,:,:]
        xt1 = warp(x0,Flow_t_0)
        xt2 = warp(x1,Flow_t_1)
        temp = torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
        Mask = torch.sigmoid(self.Mask(temp,pos))
        w1, w2 = (1-t)*Mask[:,0:1,:,:], t*Mask[:,1:2,:,:]
        output = (w1*xt1+w2*xt2)/(w1+w2+1e-8)

        return output

    def forward(self, input0, input1, pos, t=0.5):

        output = self.process(input0,input1,pos,t)
        compose = torch.cat((input0, input1, output),1)
        final = self.final(compose,pos)+output
        #final = final.clamp(0,1)

        return final


class Gen(nn.Module):
    """
    Generator from WGAN
    """
    def __init__(self, wf=4):
        """
        Args:
          wf: Factor to multiply the number of filters in the covolution
        """
        super(Gen, self).__init__()
        
        self.generator = Net(wf=4)

    def forward(self, x0, x1, t=0.5):
        """
          t: Value inside the interval [0,1] that represents the placement between the two input images where the interpolated image should be synthesized 
        """
        return self.generator(x0, x1, t)
    
class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.MSE_Loss = torch.nn.MSELoss()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def triple_channel(self, x):
        """Triple color channel as it is seen as RBG.
        """ 
        if x.ndim > 3:
            return x.expand(-1, 3, -1, -1)
        else:
            return x.expand(3, -1, -1)

    def features(self, X):
        # X = torch.cat([X, X, X], dim=1)
        X = self.triple_channel(X)
        #X = normalize_batch(X)
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    
    def forward(self,X,y):

        features_y = self.features(y)
        features_x = self.features(X)
            
        # pl1_loss =   self.MSE_Loss(features_x.relu1_2, features_y.relu1_2)
        # pl2_loss =   self.MSE_Loss(features_x.relu2_2, features_y.relu2_2)
        # pl3_loss =   self.MSE_Loss(features_x.relu3_3, features_y.relu3_3)
        pl4_loss =   self.MSE_Loss(features_x.relu4_3, features_y.relu4_3)
        return pl4_loss
    
class Vgg16_1ch(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_1ch, self).__init__()
        
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        old_weights = vgg_pretrained_features[0].weight
        new_weights = torch.mean(old_weights,dim=1).unsqueeze(1)
        vgg_pretrained_features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Change input channels to 1
        vgg_pretrained_features[0].weight = torch.nn.Parameter(new_weights)
        
        self.MSE_Loss = torch.nn.MSELoss()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def features(self, X):
        #X = normalize_batch(X)
        a = self.slice1(X)
        h_relu1_2 = a
        b = self.slice2(a)
        h_relu2_2 = b
        c = self.slice3(b)
        h_relu3_3 = c
        d = self.slice4(c)
        h_relu4_3 = d
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    
    def forward(self,X,y, plot = False):

        features_y = self.features(y)
        features_x = self.features(X)
        
        if plot == True:
            cols=8
            feat_images = features_x.relu1_2.cpu().squeeze()
            #normalize to 0-1
            feat_images = (feat_images - feat_images.min())/(feat_images.max()-feat_images.min())        
        
            rows = feat_images.shape[0]//cols
            cont=0
            
            s_dim = feat_images[0].shape[-1]
            plt.figure(figsize=(3*cols,3*rows))
            for row in range(1,rows+1):
                for col in range(1,cols+1):
                    cont+=1
                    plt.subplot(rows,cols,cont)
                    plt.imshow(feat_images[cont-1][0:s_dim,:],'gray')
                    plt.axis('off')
            plt.show()
            
        # pl1_loss =   self.MSE_Loss(features_x.relu1_2, features_y.relu1_2)
        # pl2_loss =   self.MSE_Loss(features_x.relu2_2, features_y.relu2_2)
        # pl3_loss =   self.MSE_Loss(features_x.relu3_3, features_y.relu3_3)
        pl4_loss =   self.MSE_Loss(features_x.relu4_3, features_y.relu4_3)
        return pl4_loss
