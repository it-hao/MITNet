import os
import glob
from data import srdata

class Benchmark(srdata.SRData):
    def __init__(self, args, train=False):
        super(Benchmark, self).__init__(args, train)

    def _scan(self):
        hazy_apath = os.path.join(self.apath, 'val_indoor/hazy')  

        hazy_filelist = sorted(
            glob.glob(os.path.join(hazy_apath, '*' + self.ext))
        )

        return hazy_filelist

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.args.data_val)
        self.ext = '.png' 
