import os
import glob
from data import srdata

# Indoor Training Set (ITS) of RESIDE 
class Dense(srdata.SRData):
    def __init__(self, args, train=True):
        super(Dense, self).__init__(args, train)
        self.args = args
        self.repeat = 100
        self.idx_scale = 0

    def _scan(self):
        lq_apath = os.path.join(self.apath, 'hazy') 

        lq_filelist = sorted(
            glob.glob(os.path.join(lq_apath, '*' + self.ext))
        )

        return lq_filelist

    def _set_filesystem(self, dir_data):
        if self.train: 
            self.apath = dir_data + '/' + self.args.data_train  + '/train_dense'  
        else:
            self.apath = dir_data + '/' + self.args.data_train  + '/val_dense'  
        self.ext = '.png'

    def __len__(self):
        if self.train: 
            return len(self.images_lq) * self.repeat
        else:
            return len(self.images_lq)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_lq)
        else:
            return idx

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

