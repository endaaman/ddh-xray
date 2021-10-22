import numpy as np
import torch
from . import Yolor

if __name__ == '__main__':
    model = Yolor()
    t = torch.ones([2, 3, 512, 512]).type(torch.FloatTensor)
    y = model(t)
    print(len(y))
    print(y[0].shape)
