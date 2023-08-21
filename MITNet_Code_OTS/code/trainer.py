import os
from decimal import Decimal
import numpy as np
import utility
import torch.nn.functional as F
import torch
from tqdm import tqdm
from model.mutual_info import Mutual_info_reg

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.mutual1 =  Mutual_info_reg(80, 80).to('cuda')
        self.mutual2 =  Mutual_info_reg(40, 40).to('cuda')
        self.mutual3 =  Mutual_info_reg(20, 20).to('cuda')
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            self.scheduler.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'scheduler.pt'))
            )
            # for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        # for testing
        print("lr", self.scheduler.get_last_lr()[0], self.optimizer.param_groups[0]['lr'])

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (hazy, gt, _, _) in enumerate(self.loader_train):
            hazy, gt = self.prepare([hazy, gt])

            timer_data.hold()
            timer_model.tic()

            # Zero the parameter gradients 
            self.optimizer.zero_grad()
           
            # ================================================ loss ============================================
            # ==================================================================================================
            out_1, out_1_amp, out_1_phase, out_2, image_phase, pha_feas, amp_feas = self.model(hazy, 0)
            
            gt_fft = torch.fft.rfft2(gt, norm='backward')
            label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
            
            out_2_fft = torch.fft.rfft2(out_2, norm='backward')
            out_2_fft = torch.stack((out_2_fft.real, out_2_fft.imag), -1)
            gt_amp = torch.abs(gt_fft)
            gt_1 = torch.fft.irfft2(gt_amp*torch.exp(1j*image_phase), norm='backward')

            pix_loss = self.loss(out_1, gt_1) + self.loss(out_2, gt)

            fft_loss = (self.loss(out_1_amp, gt_amp) + self.loss(out_2_fft, label_fft)) * 0.05

            mutual_loss = (self.mutual1(pha_feas[-1], amp_feas[-1]) + self.mutual2(pha_feas[-2], amp_feas[-2]) + \
                self.mutual3(pha_feas[-3], amp_feas[-3])) * 0.01

            loss = pix_loss + fft_loss + mutual_loss
            # ================================================loss ============================================
            # ==================================================================================================

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print("break...")
                exit()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    16000,
                    loss.item(),
                    pix_loss.item(),
                    fft_loss.item(),
                    mutual_loss.item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            if batch + 1 == 50:
                break
                
        self.loss.end_log(16000)
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()

    def test(self):
        epoch = self.scheduler.last_epoch

        self.ckp.write_log('\nBegin Evaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.ckp.write_log('#####################[dataset={}]---[model={}]#####################'.format(self.args.data_val, self.args.model))
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                psnr_list = []
                # ssim_list = []
                for batch, (hazy, gt, hazy_filename, _) in enumerate(tqdm_test):
                    hazy, gt = self.prepare([hazy, gt])

                    out_1, out_1_amp, out_1_phase, out_2, image_phase,  pha_feas, amp_feas = self.model(hazy, 0)
                    restore = out_2

                    # --- Calculate the average PSNR --- 
                    # --- 需要保证restore和gt图像的尺寸大小一致， 这部分我们将在数据处理部分就提前处理好---
                    psnr_list.extend(utility.to_psnr(restore, gt))
                    # --- Calculate the average PSNR ---
                    # --- 该过程比较耗时，所以在验证时候，并未计算---
                    # ssim_list.extend(utility.to_ssim_skimage(restore, gt))

                    # 不保存结果也会大大减少时间，所以默认也不保存时间
                    if self.args.save_results: 
                        self.ckp.save_image(restore, hazy_filename) 
            
            self.ckp.log[-1, idx_scale] = sum(psnr_list) / len(psnr_list) # 只保存 psnr 的值
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.args.data_val,
                    scale,
                    self.ckp.log[-1, idx_scale],
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs + 1

    def adjust(self, init, fin, step, fin_step):
        if fin_step == 0:
            return  fin
        deta = fin - init
        adj = min(init + deta * step / fin_step, fin)
        return adj
