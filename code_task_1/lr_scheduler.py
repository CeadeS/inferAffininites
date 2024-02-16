import bisect
import math
import warnings
from functools import partial

from torch.optim import lr_scheduler as _lr_scheduler

def CosineAnnealingWithWarmup(total_epochs: int,
                              warmup_epochs: int,
                              min_lr: float = 0,
                              last_epoch: int = -1):
    if last_epoch == 0:
        warnings.warn("last_epoch is set to 0, is it intended?", DeprecationWarning)
    return partial(_CosineAnnealingWithWarmup, **locals())


class _CosineAnnealingWithWarmup(_lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 total_epochs: int,
                 warmup_epochs: int,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        super(_CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        def _warmup(warmup_epochs: int):

            def f(epoch):
                return (epoch + 1) / warmup_epochs

            return f

        warmup = _warmup(self.warmup_epochs)
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * warmup(self.last_epoch) for base_lr in self.base_lrs]

        else:
            new_epoch = self.last_epoch - self.warmup_epochs
            return [self.min_lr + (base_lr - self.min_lr) *
                    (1 + math.cos(math.pi * new_epoch / (self.total_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]