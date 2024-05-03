
import torch
import torch.nn as nn
from dataset import *
from utils import *
import math
import torch.optim as optim
import torch.nn.functional as F
from classifier import *
from train import *
from DDPM import *


c, h, w = 1, 28, 28 

device = torch.device("cuda")

def schedulerLR(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 15):

    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr



def findBestReconstruction(x):

  y = torch.Tensor(size=[x.shape[0]])

  for i in range(x.shape[0]):
    y[i] = (x[i].sum().item())

  return torch.argmin(y).item()


def reconstruction_module(model, data, lr=25, rec_iter = 4, rand_initi = 5):
    loss_fn = nn.MSELoss()

    loss_fn_ = nn.MSELoss(reduction='none') 

    data = data[None,:,:,:].repeat(rand_initi,1,1,1).to(device)


    z_hat = torch.randn(size=[rand_initi,c,h,w]).to(device)
    z_hat = z_hat.requires_grad_()

    cur_lr = lr
    
    optimizer = optim.Adam([z_hat], lr=0.02, betas=(0.5, 0.999))
    
    for _ in range(rec_iter):
      
      optimizer.zero_grad()
      fake_images = sample(model,z_hat.shape[0],z_hat)

      recon_loss = loss_fn_(fake_images, data)                                                          
      reconstruct_loss = loss_fn(fake_images, data)
            
      reconstruct_loss.backward()            
      optimizer.step()

 
    
      cur_lr = schedulerLR(optimizer, cur_lr, rec_iter=rec_iter)

    z_recons = z_hat.cpu().detach().clone()
    z_gen = fake_images.cpu().detach().clone()

    return  z_gen, recon_loss, z_recons , reconstruct_loss.item()


def reconstruction_pipeline(advdataset, diffusionModel,reciter = 4, randiniti = 5):

    bestReconstructions = torch.Tensor(size=[advdataset.shape[0],c,h,w])
    z_ = torch.Tensor(size=[advdataset.shape[0],c,h,w])
    
    # Used to create the ROC curve
    recon_error = torch.Tensor(size=[advdataset.shape[0],1]) 
    
    for i in tqdm(range(advdataset.shape[0])):
         
         x , recon, z_hat, recon_error[i]= reconstruction_module(diffusionModel,advdataset[i], rand_initi=randiniti, rec_iter=reciter)

         best_reconstruced_img = findBestReconstruction(recon)
         bestReconstructions[i] = x[best_reconstruced_img]
         z_[i] = z_hat[best_reconstruced_img]


    return bestReconstructions, z_
