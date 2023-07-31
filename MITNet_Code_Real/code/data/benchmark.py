import os
import glob
from data import srdata

class Benchmark(srdata.SRData):
    def __init__(self, args, train=False):
        super(Benchmark, self).__init__(args, train)

    def _scan(self):
        # for entry in os.scandir(self.dir_hr):
        #     filename = os.path.splitext(entry.name)[0]
        #     list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
        hazy_apath = os.path.join(self.apath, 'val_indoor/hazy')  # /home/zhao/hao/dehazing/ITS/val_indoor/hazy/_.img
        # hazy_apath = os.path.join(self.apath, 'val_dense/hazy') # /home/zhao/hao/dehazing/Dense/val_dense/hazy/_.img
        # hazy_apath = os.path.join(self.apath, 'val_NH/hazy')    # /home/zhao/hao/dehazing/NH/val_NH/hazy/_.img

        hazy_filelist = sorted(
            glob.glob(os.path.join(hazy_apath, '*' + self.ext))
        )

        return hazy_filelist

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.args.data_val) # /home/zhao/hao/dehazing/ITS
        self.ext = '.png' 
