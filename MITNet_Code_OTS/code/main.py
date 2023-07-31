import torch
import utility
import data
import model
import loss
from option import args

from trainer import Trainer

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if __name__ == '__main__':
    if checkpoint.ok:
        loader = data.Data(args)

        model = model.Model(args, checkpoint)

        checkpoint.write_log(
            'Total Param are {}'.format(print_network(model))
        )
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

