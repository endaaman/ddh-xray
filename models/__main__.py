import numpy as np
import torch
from . import Yolor, YOLOv4, yolor_loss

if __name__ == '__main__':
    device = 'cuda:0'
    # model = Yolor()
    model = Yolor()
    model.to(device)
    model.train()
    x = torch.ones([1, 3, 512, 512]).type(torch.FloatTensor)
    t = torch.zeros([6, 6]).type(torch.FloatTensor)
    p = model(x.to(device))
    print(len(p))
    print(p[0].shape)

    # print(p[1][0].shape)
    # print(p[1][1].shape)
    # print(p[1][2].shape)
    loss = yolor_loss(p, t.to(device), model)
    print(loss)
