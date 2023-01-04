import os
import re
import sys
import io
import json
import yaml

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import ticker, pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from effdet import DetBenchTrain, DetBenchPredict
from ptflops import get_model_complexity_info
from timm.scheduler import CosineLRScheduler

from endaaman import get_paths_from_dir_or_file
from endaaman.torch import TorchCommander, Trainer, Predictor

from datasets import XRBBDataset, LABEL_TO_STR, IMAGE_STD, IMAGE_MEAN
from utils import get_state_dict
from models import create_det_model, yolor_loss, SIZE_BY_DEPTH
from models.ssd import MultiBoxLoss



class EffDetTrainer(Trainer):
    def prepare(self, **kwargs):
        self.bench = DetBenchTrain(self.model).to(self.device)

    def create_scheduler(self, lr):
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=5, t_initial=90,
            warmup_lr_init=lr/2, lr_min=lr/1000,
            warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def eval(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels['bbox'] = labels['bbox'].to(self.device)
        labels['cls'] = labels['cls'].to(self.device)
        loss = self.bench(inputs, labels)
        return loss['loss'], None

    def get_metrics(self):
        return {
            'batch': { },
            'epoch': { },
        }


def select_best_bbs(bbs):
    '''
    bbs: [[x0, y0, x1, y1, cls]]
    '''
    best_bbs = []
    missing = []
    for label, text in LABEL_TO_STR.items():
        m = bbs[bbs[:, 4].long() == label]
        if len(m) > 0:
            # select first bb
            best_bbs.append(m[0])
        else:
            missing.append(text)
    return torch.stack(best_bbs), ' '.join(missing)


class EffDetPredictor(Predictor):
    def prepare(self, **kwargs):
        self.bench = DetBenchPredict(self.model).to(self.device)
        m = re.match(r'.*_(d\d)$', self.checkpoint.name)
        assert m
        self.image_size = SIZE_BY_DEPTH[m[1]]

    def create_model(self):
        model = create_det_model(self.checkpoint.name)
        model.load_state_dict(self.checkpoint.model_state)
        return model.eval()

    def eval(self, inputs):
        return self.bench(inputs.to(self.device)).detach().cpu()

    def collate(self, pred, idx):
        bbs = pred
        bbs[:, 4] = bbs[:, 5]
        bbs, missing = select_best_bbs(bbs)
        if len(missing) > 0:
            print(f'[{idx}] missing: {missing}')
        # bbs: [[x0, y0, x1, y1, cls, index]]
        bbs[:, 5] = idx
        return bbs

    def start(self, images):
        scales = [torch.tensor(i.size)/self.image_size for i in images]
        images = [i.resize((self.image_size, self.image_size)) for i in images]

        bbss = super().start(images)
        for bbs, scale in zip(bbss, scales):
            scale = scale.repeat_interleave(2)
            bbs[:, :4] *= scale
        return bbss


class CMD(TorchCommander):
    def create_loaders(self, mode, size, collate_fn=None):
        if mode not in ['effdet', 'yolo', 'ssd']:
            raise ValueError(f'Invalid target: {mode}')

        return [
            self.as_loader(
                XRBBDataset(mode=mode, target=target, size=size),
                collate_fn=collate_fn
            ) for target in ['train', 'test']
        ]

    def arg_train(self, parser):
        parser.add_argument('-d', '--depth', default='d0', choices=list(SIZE_BY_DEPTH.keys()))

    def run_train(self):
        name = f'effdet_{self.a.depth}'
        model = create_det_model(name)
        size = SIZE_BY_DEPTH[self.a.depth]
        loaders = self.create_loaders(mode='effdet', size=size)

        trainer = self.create_trainer(
            T=EffDetTrainer,
            name=name,
            model=model,
            loaders=loaders,
        )

        self.start(trainer)


    def draw_bbs(self, imgs, bbss):
        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-L.ttf', 20)
        results = []
        for img, bbs in zip(imgs, bbss):
            draw = ImageDraw.Draw(img)
            for _, bb in enumerate(bbs):
                label = bb[4].item()
                bbox = bb[:4]
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)
                draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=font, fill='yellow')
            results.append(img)
        return results

    def arg_predict(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--src', '-s', required=True)

    def run_predict(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)

        predictor = self.create_predictor(
            P=EffDetPredictor,
            checkpoint=checkpoint,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD)

        paths = get_paths_from_dir_or_file(self.a.src)
        images = [Image.open(p) for p in paths]

        bbss = predictor.start(images=images)

        results = self.draw_bbs(images, bbss)

        dest_dir = os.path.join('out', checkpoint.name, 'predict')
        os.makedirs(dest_dir, exist_ok=True)
        for result, path in zip(results, paths):
            name = os.path.splitext(os.path.basename(path))[0]
            result.save(os.path.join(dest_dir, f'{name}.jpg'))

        print('done')

    def arg_map(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    def run_map(self):
        ds = XRBBDataset(mode='effdet', target=self.a.target, size=self.image_size)

        images = [i.image for i in ds.items]
        bbss = self.detect_rois(images, self.image_size)

        for (bbs, item) in zip(bbss, ds.items):
            print(bbs)
            print(item.bb)
            self.bbs = bbs
            self.item = item
            break


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 100,
        'lr': 0.01,
        'batch_size': 8,
        'save_period': 25,
    })
    cmd.run()
