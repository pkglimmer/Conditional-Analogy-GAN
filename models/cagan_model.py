import torch
import itertools
from torch._C import device
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from .image_pool import ImagePool
from .base_model import BaseModel
from . import cagan_networks
import os


class CAGANModel(BaseModel):
    """
    This class implements the CAGANGAN model
    modified from cycle_gan_model.py in original CycleGAN repo
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True, dataset_mode='triplet')  # default CycleGAN did not use dropout
        return parser

    def __init__(self, opt):
        """Initialize the CAGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G', 'cycle', 'idt', 'G_D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names =  ['xi', 'yi', 'yj', 'xij','alpha', 'xij1', 'rec_xi'] # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G', 'D']

        # define networks (Generator and discriminator)
        self.netG = cagan_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  
            # define discriminator
            self.netD = cagan_networks.define_D(opt.D_input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.fake_ouput_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.rec_input_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionGAN = cagan_networks.GANLoss('bce').to(self.device)  # vanilla: BCEwithlogits loss, basic: BCELoss(cuz sigmoid already done in netD)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.image_paths, input_tensor = input
        self.xi, self.yi, self.yj = [Variable(t.to(self.device)) for t in input_tensor]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_output = self.netG(torch.cat([self.xi, self.yi, self.yj], dim=1))

        self.alpha = self.fake_output[:,0:1,:,:] 
        self.xij1 = self.fake_output[:,1:,:,:]
        self.xij = self.alpha * self.xij1 + (1-self.alpha) * self.xi # [64, 3, 128, 96]
        
        self.rec_input = self.netG(torch.cat([self.xij, self.yj, self.yi], dim=1))   
        self.rec_alpha = self.rec_input[:,0:1,:,:]
        self.rec_xi = self.rec_input[:,1:,:,:]
        self.rec_xi = self.rec_alpha * self.rec_xi + (1-self.rec_alpha) * self.xij


    def backward_D(self): 
        """Calculate GAN loss for the discriminator

        One positive term: cat(xi, yi)
        Two negative terms: cat(xij, yj), cat(xi, yj)
        Because xij will be used in netG back propagation and is not used  in D optimization, it HAS TO BE detached.
        """
        # pred 
        pred_pos = self.netD(torch.cat([self.xi, self.yi], dim=1))
        pred_neg1 = self.netD(torch.cat([self.xij.detach(), self.yj], dim=1))
        pred_neg2 = self.netD(torch.cat([self.xi, self.yj], dim=1))

        # loss
        loss_D_pos = self.criterionGAN(pred_pos, 1) #param(BCEwithlogits): logits,target
        loss_D_neg1 = self.criterionGAN(pred_neg1, 0)
        loss_D_neg2 = self.criterionGAN(pred_neg2, 0) #length1 tensor
        # Combined loss and calculate gradients
        self.loss_D = loss_D_pos + loss_D_neg1 + loss_D_neg2
        self.loss_D.backward()
        return self.loss_D

    def backward_G(self):
        """Calculate the loss for generator"""
        gamma_i = 0.1 # identity loss factor
        gamma_c = 1.0 # cycle loss factor

        # Identity loss
        self.loss_idt = torch.mean(torch.abs(self.alpha))

        # GAN loss  
        pred_neg1 = self.netD(torch.cat([self.xij, self.yj], dim=1))
        # self.loss_G_D = self.criterionGAN(pred_neg1, torch.ones_like(pred_neg1))
        self.loss_G_D = self.criterionGAN(pred_neg1, 1)

        # cycle loss || G(G(xi,yi,yj),yj,yi) - xi||
        self.loss_cycle = self.criterionCycle(self.rec_xi, self.xi) 

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_D + self.loss_cycle * gamma_c + self.loss_idt * gamma_i
        self.loss_G.backward()
        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()      # compute fake images and reconstruction images.

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

