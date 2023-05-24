"""
Author: Duy-Phuong Dao
Email : phuongdd.1997@gmail.com or duyphuongcri@gmail.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torch.nn import init
import math 
from models.swin_backbone import SwinEncoder, SwinDecoder
from models.resnet_backbone import ResNetEncoder
from config import model_config

class MVAE(nn.Module):
    def __init__(self, config):
        super(MVAE, self).__init__()
        self.config=config

        self.z_mean_mri = nn.Linear(config.LATENT_IMG_DIM, config.Z_DIM)
        self.z_log_var_mri = nn.Linear(config.LATENT_IMG_DIM, config.Z_DIM)

        self.z_mean_pet = nn.Linear(config.LATENT_IMG_DIM, config.Z_DIM)
        self.z_log_var_pet = nn.Linear(config.LATENT_IMG_DIM, config.Z_DIM)

        self.z_mean_dti = nn.Linear(config.LATENT_IMG_DIM, config.Z_DIM)
        self.z_log_var_dti = nn.Linear(config.LATENT_IMG_DIM, config.Z_DIM)

        self.experts = ProductOfExperts()
        self.z_resample = nn.Linear(config.Z_DIM, config.LATENT_IMG_DIM)

        if config.BACKBONE_VAE == "Swin":
            self.encoder_mri = SwinEncoder(img_size=self.config.IMAGE_SIZE, in_channels=1, feature_size=self.config.IMAGE_BASE_DIMEN)
            self.decoder_mri = SwinDecoder(feature_size=self.config.IMAGE_BASE_DIMEN, out_channels=1)

            self.encoder_pet = SwinEncoder(img_size=self.config.IMAGE_SIZE, in_channels=1, feature_size=self.config.IMAGE_BASE_DIMEN)
            self.decoder_pet = SwinDecoder(feature_size=self.config.IMAGE_BASE_DIMEN, out_channels=1)

            self.encoder_dti = SwinEncoder(img_size=self.config.IMAGE_SIZE, in_channels=2, feature_size=self.config.IMAGE_BASE_DIMEN)
            self.decoder_dti = SwinDecoder(feature_size=self.config.IMAGE_BASE_DIMEN, out_channels=2)
        elif config.BACKBONE_VAE == "ResNet":
            self.encoder_mri = ResNetEncoder(in_channels=1, feature_size=self.config.IMAGE_BASE_DIMEN)
            self.decoder_mri = SwinDecoder(feature_size=self.config.IMAGE_BASE_DIMEN, out_channels=1)

            self.encoder_pet = ResNetEncoder(in_channels=1, feature_size=self.config.IMAGE_BASE_DIMEN)
            self.decoder_pet = SwinDecoder(feature_size=self.config.IMAGE_BASE_DIMEN, out_channels=1)

            self.encoder_dti = ResNetEncoder(in_channels=2, feature_size=self.config.IMAGE_BASE_DIMEN)
            self.decoder_dti = SwinDecoder(feature_size=self.config.IMAGE_BASE_DIMEN, out_channels=2)
            
        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def prior_expert(self, size):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).

        @param size: integer
                    dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                        cast CUDA on variables
        """
        mu     = Variable(torch.zeros(size)).to(self.config.DEVICE)
        logvar = Variable(torch.log(torch.ones(size))).to(self.config.DEVICE)
        # if use_cuda:
        #     mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar
        
    def infer(self, data, mask): 
        """
        data: B x C x D x H x W
        mask: B x 3
        """
        B = data.shape[0]

        # use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar         = self.prior_expert((1, B, self.config.Z_DIM))
        mri_mu, mri_logvar = self.prior_expert((B, self.config.Z_DIM))
        pet_mu, pet_logvar = self.prior_expert((B, self.config.Z_DIM))
        dti_mu, dti_logvar = self.prior_expert((B, self.config.Z_DIM))
        ###########
        mask_non_misssing = torch.ones((1, B), dtype=torch.bool).to(self.config.DEVICE)
        # if use_cuda:
        #     mask_non_misssing = mask_non_misssing.cuda()


        if mask[:, 0].sum() != 0: # Check if MRI exists in the batch or not
            mri =  torch.flatten(self.encoder_mri(data[:, :1]), start_dim=1)
            mri_mu, mri_logvar = self.z_mean_mri(mri), self.z_log_var_mri(mri)
            mu     = torch.cat((mu, mri_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, mri_logvar.unsqueeze(0)), dim=0)
            mask_non_misssing = torch.cat((mask_non_misssing, mask[:, 0].unsqueeze(0)), dim=0)
        ### PET
        if mask[:, 1].sum() != 0: # Check if PET exists in the batch or not
            pet =  torch.flatten(self.encoder_pet(data[:, 1:2]), start_dim=1)
            pet_mu, pet_logvar = self.z_mean_pet(pet), self.z_log_var_pet(pet)
            mu     = torch.cat((mu, pet_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, pet_logvar.unsqueeze(0)), dim=0)
            mask_non_misssing = torch.cat((mask_non_misssing, mask[:, 1].unsqueeze(0)), dim=0)
        ### DTI

        if mask[:, 2].sum() != 0: # Check if DTI exists in the batch or not
            dti =  torch.flatten(self.encoder_dti(data[:, 2:]), start_dim=1)
            dti_mu, dti_logvar = self.z_mean_dti(dti), self.z_log_var_dti(dti)
            mu     = torch.cat((mu, dti_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, dti_logvar.unsqueeze(0)), dim=0)
            mask_non_misssing = torch.cat((mask_non_misssing, mask[:, 2].unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        if not self.config.PRIOR_EXPERT:
            mu, logvar, mask_non_misssing = mu[1:], logvar[1:], mask_non_misssing[1:]
        #### Merge multiple distribution
        mu, logvar = self.experts(mu, logvar, mask_non_misssing)

        return mu, logvar, mri_mu, mri_logvar, pet_mu, pet_logvar, dti_mu, dti_logvar 

    def forward(self, data, mask):
        """
        Input should be a single timepoint
        data  :  B x M x D x H x W
        mask  :  B x M 
        mu    :  M x B x D_z
        logvar:  M x B x D_z
        """
        B = data.shape[0]

        mu, logVar, mri_mu, mri_logVar, pet_mu, pet_logVar, dti_mu, dti_logVar  = self.infer(data, mask)
        z = self.reparameterize(mu, logVar)
        z = self.z_resample(z)
        z = z.view([B, self.config.IMAGE_LATENT_DIM]+self.config.LATENT_IMG_SIZE)

        mri_hat = self.decoder_mri(z)
        pet_hat = self.decoder_pet(z)
        dti_hat = self.decoder_dti(z)
        out_hat = torch.cat((mri_hat, pet_hat, dti_hat), dim=1)
        return out_hat, mu, logVar, mri_mu, mri_logVar, pet_mu, pet_logVar, dti_mu, dti_logVar
    
class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x B x D for M experts
    @param logvar: M x B x D for M experts
    mask : M x B
    """
    def forward(self, mu, logvar, mask, eps=1e-8):
        # mask = torch.transpose(mask, 1, 0) # -> M x B
        mask = mask.unsqueeze(-1) # -> M x B x 1
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T * mask, dim=0) / torch.sum(T * mask, dim =0)
        pd_var    = 1. / torch.sum(T * mask, dim =0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar
    

