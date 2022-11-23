import os
import sys
import io
import json
import yaml
import time
import datetime

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import ticker, pyplot as plt
import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from effdet.efficientdet import HeadNet
import yolov5
from ptflops import get_model_complexity_info
from yolov5.models.yolo import Model as Yolo5

from endaaman.torch import Trainer

from datasets import ROIDataset
from utils import get_state_dict
from models import YOLOv3, SSD300, Yolor, yolor_loss
from models.ssd import MultiBoxLoss


SIZE_BY_DEPTH = {
    'd0': 128 * 4,
    'd1': 128 * 5,
    'd2': 128 * 6,
    'd3': 128 * 7,
    'd4': 128 * 8,
    'd5': 128 * 10,
    'd6': 128 * 12,
    'd7': 128 * 14,
}


class T(Trainer):
    def arg_common(self, parser):
        pass
        # parser.add_argument('--no-aug', action='store_true')

    def create_model(self, model_name, sub_name=None):
        if model_name == 'yolo':
            model = YOLOv3()
        elif model_name == 'yolor':
            model = Yolor(num_classes=7)
        elif model_name == 'yolo5':
            # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model = Yolo5(cfg='cfg/yolov5s.yaml')
        elif model_name == 'effdet':
            assert sub_name
            cfg = get_efficientdet_config(f'tf_efficientdet_{sub_name}')
            cfg.num_classes = 6
            model = EfficientDet(cfg)
        elif model_name == 'ssd':
            model = SSD300(n_classes=7)
        else:
            raise ValueError(f'Ivalid model_name: {model_name}')

        return model.to(self.device).train()

    def create_loaders(self, mode, image_size, collate_fn=None):
        if mode not in ['effdet', 'yolo', 'ssd']:
            raise ValueError(f'Invalid target: {mode}')

        loaders = [
            self.as_loader(
                ROIDataset(mode=mode, target=target),
                collate_fn=collate_fn
            ) for target in ['train', 'test']
        ]
        return loaders


    def arg_effdet(self, parser):
        parser.add_argument('-d', '--depth', default='d0', choices=list(SIZE_BY_DEPTH.keys()))

    def run_effdet(self):
        depth = self.args.depth
        model = self.create_model('effdet', depth)
        bench = DetBenchTrain(model).to(self.device)
        loaders = self.create_loaders('effdet', SIZE_BY_DEPTH[self.args.depth])

        def eval_fn(inputs, labels):
            inputs = inputs.to(self.device)
            labels['bbox'] = labels['bbox'].to(self.device)
            labels['cls'] = labels['cls'].to(self.device)
            loss = bench(inputs, labels)
            return loss['loss'], None

        self.train_model(
            name=f'effdet_{depth}',
            model=model,
            train_loader=loaders[0],
            val_loader=loaders[0],
            eval_fn=eval_fn,
            no_metrics=True,
        )

    def run_yolo(self):
        model = self.create_model('yolo')
        loaders = self.create_loaders('yolo', 512)

        def eval_fn(inputs, labels):
            for idx, ll in enumerate(labels):
                ll[:, 0] = idx
            inputs = inputs.to(self.device)
            labels = labels.view(-1, 6).to(self.device) # batch x [batch_idx, cls_id, x, y, w, h]
            loss, outputs = model(inputs, labels)
            return loss, outputs

        def save_hook(weights_dir, weights_name, weights):
            name = weights['epoch'] + '.darknet'
            model.save_darknet_weights(os.path.join(weights_dir, name))
            weights['darknet_weight'] = name
            return weights_dir, weights_name, weights

        # self.train_model(
        #     model,
        #     loaders,
        #     eval_fn, {
        #         # metrics_fn
        #     },
        #     # save_hook
        # )

    def run_yolor(self):
        model = self.create_model('yolor')
        loaders = self.create_loaders('yolo', 512)

        def eval_fn(inputs, labels):
            for idx, ll in enumerate(labels):
                ll[:, 0] = idx
            inputs = inputs.to(self.device)
            labels = labels.view(-1, 6).to(self.device) # batch x [batch_idx, cls_id, x, y, w, h]
            outputs = model(inputs)
            loss = yolor_loss(outputs, labels, model)
            return loss[0], outputs

        # self.train_model(
        #     model,
        #     loaders,
        #     eval_fn, {
        #         # metrics_fn
        #     },
        # )

    def run_ssd(self):
        def collate_fn(batch):
            xx = []
            yy = []
            for (x, y) in batch:
                xx.append(x)
                yy.append(y)
            return torch.stack(xx), yy

        model = self.create_model('ssd')
        loaders = self.create_loaders('ssd', 300, collate_fn)
        criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)

        def eval_fn(inputs, labels):
            inputs = inputs.to(self.device)
            bb = []
            cc = []
            for b, c in labels:
                bb.append(b.to(self.device))
                cc.append(c.to(self.device))
            predicted_locs, predicted_scores = model(inputs)
            loss = criterion(predicted_locs, predicted_scores, bb, cc)
            if float(loss.item()) > 1000000:
                vv = [predicted_locs, predicted_scores, bb, cc]
                vvv = []
                for v in vv:
                    if torch.is_tensor(v):
                        vvv.append(v.to('cpu'))
                    else:
                        vvv.append([a.to('cpu') for a in v])
                torch.save([*vv, loss.item()], 'tmp/loss.pth')
                print(loss.item())
                sys.exit(0)
            return loss, None

        # self.train_model(
        #     model,
        #     loaders,
        #     eval_fn, {
        #         # metrics_fn
        #     })

    def arg_fake(self, parser):
        parser.add_argument('-m', '--model-name', required=True)
        parser.add_argument('-d', '--depth', default='d0', choices=list(SIZE_BY_DEPTH.keys()))

    def run_fake(self):
        model = self.create_model('effdet', self.args.depth)
        weights_path = self.save_weights(model, 0, {}, {})
        print(weights_path)

    def run_flops(self):
        for model_name in ['yolo', 'ssd', 'yolo5']:
            model = self.create_model(model_name)
            flops, params = get_model_complexity_info(
                model,
                (3, 512, 512, ),
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False,
            )
            flops = flops[:-5]
            params = params[:-2]
            batch = 1

            inputs = torch.ones(3, 3, 512, 512).to('cuda')
            model.to('cuda')
            scale = 1000

            # START
            starting_time = time.perf_counter()
            for i in range(scale):
                _ = model(inputs)
            elapsed_time = time.perf_counter() - starting_time
            # END
            duration = elapsed_time / scale

            print(f'{model_name}: {params}M param {flops}G flops {duration*1000}ms')

            del model



if __name__ == '__main__':
    t = T({
        'epoch': 100,
        'lr': 0.001,
        'batch_size': 32,
    })
    t.run()
