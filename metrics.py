import torch


def calc_acc(x, y, thresh_hold=0.5):
    x = x.view(-1) > thresh_hold
    y = y.view(-1) > thresh_hold
    return (x == y).sum().item() / len(x)

def calc_recall(x, y, thresh_hold=0.5):
    x = x.view(-1) > thresh_hold
    y = y.view(-1) > thresh_hold
    tp = x * y
    p = (y == 1).sum().item()
    return tp.sum().item() / p if p > 0 else 0

def calc_spec(x, y, thresh_hold=0.5):
    x = x.view(-1) > thresh_hold
    y = y.view(-1) > thresh_hold
    not_y = torch.logical_not(y)
    tn = torch.logical_not(x) * not_y

    n = not_y.sum().item()
    return tn.sum().item() / n if n > 0 else 0
