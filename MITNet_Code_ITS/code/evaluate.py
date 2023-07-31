import argparse
import os
import time 
import time
import torch
import torch.nn as nn

from utils import utils_logger
from utils import utils_test as utils
from torch.utils.data import DataLoader
from math import log10
from model.mitnet import MITNet as Net
from data.test_data import TestData

parser = argparse.ArgumentParser(description="Testing for Dehazing Models")
parser.add_argument("--model", type=str, default='MITNet', help="Mode name")
parser.add_argument('--pre_train', type=str, default='../ckpt/mitnet_indoor_best.pt', help="Path to pretrained model") 
parser.add_argument("--dataset", type=str, default='indoor', help='indoor, outdoor, nh, dense')
parser.add_argument("--downsample", type=int, default=16, help='maxmium downsample factor')
parser.add_argument("--batch_size", type=int, default=1, help='batch size')
parser.add_argument("--save_path", type=str, default='../experiment/test_results', help='Save restoration results')
parser.add_argument("--save_image", type=bool, default=True)
parser.add_argument("--multi_supervised", type=bool, default=True)

opt = parser.parse_args()

save_path = os.path.join(opt.save_path, opt.model + '/' + opt.dataset + '_results') 

if not os.path.exists(save_path):
        utils.create_dir(save_path)

if opt.dataset == 'indoor':
    test_data_dir = '/home/zhao/hao/dehazing/ITS/val_indoor'
    
elif opt.dataset == 'outdoor':
    test_data_dir = '/home/zhao/hao/dehazing/OTS/val_outdoor'
    
elif opt.dataset == 'dense':
    test_data_dir = '/home/zhao/hao/dehazing/Dense/val_dense'

elif opt.dataset == 'nh':
    test_data_dir = '/home/zhao/hao/dehazing/NH/val_nh'


lg = utils_logger.logger_info('efficient image dehazing', log_path=os.path.join(opt.save_path, opt.model + '_' + opt.dataset + '.log'))
lg.info("============Begin Evaluation============")
lg.info('Model: %s || dataset_name: %s|| pre_train: %s' % (opt.model, opt.dataset, opt.pre_train))

def main():
    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Validation data loader --- #
    lg.info('Loading data: %s' % (opt.dataset))
    val_data_loader = DataLoader(TestData(opt.dataset, test_data_dir, opt.downsample), \
        batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # --- Define the network --- #
    lg.info('Loading model: %s' % (opt.pre_train))
    net = Net(in_chn=3, wf=20, depth=4)

    # --- Multi-GPU --- #
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids).module

    # --- Load the network weight --- #
    net.load_state_dict(
                torch.load(
                    os.path.join(opt.pre_train),
                    **{}
                ),
                strict=False
    )

    # --- Use the evaluation model in testing --- #
    net.eval()
    lg.info('--- Testing starts! ---')
    start_time = time.time()

    val_psnr, val_ssim, median_time = utils.validation(net, val_data_loader, device, opt.save_image, save_path, opt.multi_supervised)
    end_time = time.time() - start_time
    lg.info('val_psnr: {0:.3f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
    lg.info('validation time is {0:.4f}'.format(end_time))
    lg.info('validation average time is {0:.4f}'.format(end_time / len(val_data_loader)))
    lg.info('validation median_time time is {0:.4f}'.format(median_time))
    lg.info('--- Testing end! ---')
    lg.info("============End Evaluation============\n")


if __name__ == "__main__":
    main()
