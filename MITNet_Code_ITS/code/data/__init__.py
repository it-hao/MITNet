from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            # 训练
            datasets = []
            module_name = args.data_train
            m = import_module('data.' + module_name.lower())
            datasets.append(getattr(m, module_name)(args))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        # 验证
        if args.data_val in ['ITS']:
            module_test = import_module('data.its')
            testset = getattr(module_test, "ITS")(args, train=False)
        elif args.data_val in ['OTS'] :
            module_test = import_module('data.ots')
            testset = getattr(module_test, "OTS")(args, train=False)
        elif args.data_val in ['Dense'] :
            module_test = import_module('data.dense')
            testset = getattr(module_test, "Dense")(args, train=False)
        elif args.data_val in ['NH'] :
            module_test = import_module('data.nh')
            testset = getattr(module_test, "NH")(args, train=False)
        elif args.data_val in ['Haze4K'] :
            module_test = import_module('data.haze4k')
            testset = getattr(module_test, "Haze4K")(args, train=False)

        self.loader_test = dataloader.DataLoader(
            testset,
            # batch_size=args.batch_size, # 不设置为1的话，验证的时候速度更快
            batch_size=1, # 不设置为1的话，验证的时候速度更快
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
        )

# for testing Dataloader
# import os
# import sys
# sys.path.append(os.pardir)
# from option import args

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# if __name__ == '__main__':
#     data = Data(args)
#     loader_test = data.loader_test
#     for batch, (lr, hr, filename, filename2, ) in enumerate(loader_test):
#         print(lr.size(), hr.size(), filename, filename2)
#         if batch == 1:
#             break