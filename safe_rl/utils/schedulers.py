from torch.optim.lr_scheduler import _LRScheduler

class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, end_learning_rate=1e-6, power=1.0, last_epoch=-1):
        self.max_iter = max_iter
        self.end_learning_rate = end_learning_rate
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (base_lr - self.end_learning_rate) * ((1 - min(self.last_epoch, self.max_iter) / self.max_iter) ** self.power)
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]