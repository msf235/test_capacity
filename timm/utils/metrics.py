""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_pr_dim(X, preserve_gradients=True):
    # X_centered = X - np.mean(X, axis=0)
    # C = X_centered.T @ X_centered / (X.shape[0]-1)
    N = X.shape[0]
    X_centered = X-torch.mean(X, dim=0)
    if X.shape[0] < X.shape[1]:
        X_centered = X_centered.T
    C = X_centered.T@X_centered/(N-1)
    eigs = torch.symeig(C, eigenvectors=preserve_gradients)[0]
    return torch.sum(eigs)**2/torch.sum(eigs**2)
