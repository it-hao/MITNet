import os
import imageio
import numpy as np
import torch.utils.data as data

from data import common

# --- Validation/test dataset --- #
class TestData(data.Dataset):
    def __init__(self, dataset_name, val_data_dir, downsample):
        super().__init__() 
        self.dataset_name = dataset_name
        val_list = os.path.join(val_data_dir, 'vallist.txt') 
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            if self.dataset_name=='NH' or self.dataset_name=='dense':
                gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names] #haze_names#
            elif self.dataset_name=='indoor':
                gt_names = [i.split('_')[0] + '.png' for i in haze_names] 
            elif self.dataset_name=='outdoor':
                gt_names = [i.split('_')[0] + '.jpg' for i in haze_names]   
            else:
                gt_names = None 
                print('The dataset is not included in this work.')  
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.data_list=val_list
        self.downsample = downsample

    def get_images(self, index):
        haze_name = self.haze_names[index]
        haze_img  = imageio.imread(os.path.join(self.val_data_dir, 'hazy', haze_name))
        
        gt_name = self.gt_names[index]
        gt_img  = imageio.imread(os.path.join(self.val_data_dir, 'gt', gt_name)) 

        size = haze_img.shape[:2]
        h = size[0] - np.mod(size[0], self.downsample)
        w = size[1] - np.mod(size[1], self.downsample)
        haze_img = haze_img[0:h, 0:w, :]
        gt_img = gt_img[0:h, 0:w, :]
        haze_img, gt_img = common.set_channel([haze_img, gt_img], 3)
        haze_img, gt_img = common.np2Tensor([haze_img, gt_img], 1)
                
        return haze_img, gt_img, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

