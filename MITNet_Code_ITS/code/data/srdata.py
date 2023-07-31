import os
import numpy as np
import imageio
import torch.utils.data as data

from data import common

class SRData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.scale = args.scale

        self._set_filesystem(args.dir_data)
        self.images_lq = self._scan()

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def __getitem__(self, idx):
        lq, hq, hq_filename, lq_filename = self._load_file(idx)
        lq, hq = self._get_patch(lq, hq, self.args.downsample)
        lq, hq = common.set_channel([lq, hq], self.args.n_colors)
        lq_tensor, hq_tensor = common.np2Tensor([lq, hq], self.args.rgb_range)
        return lq_tensor, hq_tensor, hq_filename, lq_filename
        
    def __len__(self):
        return len(self.images_lq)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        hazy_path = self.images_lq[idx]
        
        hazy_root, hazy_name = os.path.split(hazy_path)
        gt_root = hazy_root.replace("hazy", "gt")  

        hazy_filename = os.path.splitext(hazy_name)[0] 
        gt_filename = hazy_name.split('_')[0] 
        gt_path = os.path.join(gt_root, gt_filename + self.ext) 

        hazy_img = imageio.imread(hazy_path)
        gt_img = imageio.imread(gt_path)

        return hazy_img, gt_img, hazy_filename, gt_filename

    def _get_patch(self, lq, hq, downsample):
        patch_size = self.args.patch_size
        if self.train:
            lq, hq = common.get_patch(lq, hq, patch_size)
            lq, hq = common.augment([lq, hq])
        else:
            if 1 != downsample:
                size = lq.shape[:2]
                h = size[0] - np.mod(size[0], downsample)
                w = size[1] - np.mod(size[1], downsample)
                lq = lq[0:h, 0:w, :]
                hq = hq[0:h, 0:w, :]

        return lq, hq

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

