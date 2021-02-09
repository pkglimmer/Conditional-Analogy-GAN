import os
import torch
from .base_dataset import BaseDataset
from PIL import Image
from torchvision import transforms


class TripletDataset(BaseDataset):
    """
    This dataset class can load triplet datasets.
    xi: image of model wearing yi
    yi: image of standalone clothes yi
    yj: another standalone clothes
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_x = os.path.join(opt.dataroot, opt.phase + '_x')  
        self.dir_yi = os.path.join(opt.dataroot, opt.phase + '_yi')  
        self.dir_yj = os.path.join(opt.dataroot, opt.phase + '_yj')  

        self.x_paths = [os.path.join(self.dir_x,s) for s in sorted(os.listdir(self.dir_x))]
        self.yi_paths = [os.path.join(self.dir_yi,s) for s in sorted(os.listdir(self.dir_yi))]
        self.yj_paths = [os.path.join(self.dir_yj,s) for s in sorted(os.listdir(self.dir_yj))]

        self.dataset_size = len(self.x_paths)  # get the size of dataset 

        self.transform = transforms.Compose([
            # transforms.Resize((132, 100)), 
            # transforms.RandomCrop((128, 96)),  # basic augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_and_transform(self, paths, index):
        img_path = paths[index % self.dataset_size] 
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            xi (tensor)       -- an image of human model wearing clothes yi
            yi (tensor)       -- image tensor of clothes yj
            yj (tensor)       -- image tensor of clothes yj
       """
        x_img = self.load_and_transform(self.x_paths, index)
        yi_img = self.load_and_transform(self.yi_paths, index)
        yj_img = self.load_and_transform(self.yj_paths, index)
        x_paths = self.x_paths[index % self.dataset_size] 

        return x_paths, [x_img, yi_img, yj_img]

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.dataset_size