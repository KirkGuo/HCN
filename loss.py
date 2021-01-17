import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable


class LabelSmoothing(nn.Module):

    def __init__(self, padding_idx, smoothing=0.0, device=0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum').cuda(device)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (x.size(1) - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Scheduler:

    def __init__(self, model, pad_idx, args):
        self.criterion = LabelSmoothing(pad_idx, smoothing=args.smoothing, device=args.gpu)
        #self.criterion = nn.CrossEntropyLoss(reduction='sum').cuda(args.gpu)#(ignore_index=-1)
        self.optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay
        )
        self.warm_up = args.warm_up
        self.curr_step = 0
        self.init_lr = args.lr
        self.curr_loss = None
        self.lr_decay = args.lr_decay

    def __call__(self, x, y, norm):
        self.curr_loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return self.curr_loss.data * norm

    def step(self, epoch):
        self.curr_loss.backward()
        self._update(epoch)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _update(self, epoch):
        self.curr_step += 1
        lr = self.init_lr * self._lr_scale(epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _lr_scale(self, epoch):
        if epoch <= self.warm_up:
            return 1
        else:
            return self.lr_decay
