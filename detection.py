import os
import re
import sys
import io
import json
import yaml

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pydantic import Field, validator
import numpy as np
import matplotlib
from matplotlib import ticker, pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler
from effdet import DetBenchTrain, DetBenchPredict, EfficientDet, get_efficientdet_config
from ptflops import get_model_complexity_info
from mean_average_precision import MetricBuilder

from endaaman import load_images_from_dir_or_file
from endaaman.ml import BaseMLCLI, BaseTrainer, BaseTrainerConfig

from datasets import XRBBDataset, LABEL_TO_STR, IMAGE_STD, IMAGE_MEAN
from utils import get_state_dict
from models import SIZE_BY_DEPTH



class EffDetTrainerConfig(BaseTrainerConfig):
    depth:str
    scheduler:str = 'static'

    def size(self):
        return SIZE_BY_DEPTH[self.depth]


class EffDetTrainer(BaseTrainer):
    def prepare(self):
        # if m := re.match(r'^effdet_d(\d)$', name):
        cfg = get_efficientdet_config(f'tf_efficientdet_{self.config.depth}')
        cfg.num_classes = 6
        model =  EfficientDet(cfg)
        self.bench = DetBenchTrain(model).to(self.device)
        return model

    def create_scheduler(self):
        if m := re.match(r'^cosine_(\d+)$', self.config.scheduler):
            scale = int(m[1])
            return CosineAnnealingLR(self.optimizer, T_max=50, eta_min=self.config.lr/scale)
        return None

    def eval(self, inputs, gts):
        self.bench.to(self.device)
        inputs = inputs.to(self.device)
        gts = {
            'bbox': gts['bbox'].to(self.device),
            'cls': gts['cls'].to(self.device)
        }
        d = self.bench(inputs, gts)
        return d['loss'], None

    def get_metrics(self):
        return {}


class EffDetPredictor():
    pass


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



def calc_iou(a, b):
    a_area = (a[...,2] - a[...,0]) * (a[...,3] - a[...,1])
    b_area = (b[...,2] - b[...,0]) * (b[...,3] - b[...,1])
    x_min = torch.maximum(a[...,0], b[...,0])
    y_min = torch.maximum(a[...,1], b[...,1])
    x_max = torch.minimum(a[...,2], b[...,2])
    y_max = torch.minimum(a[...,3], b[...,3])
    w = torch.maximum(torch.tensor(.0), x_max - x_min)
    h = torch.maximum(torch.tensor(.0), y_max - y_min)
    intersect = w * h
    return intersect / (a_area + b_area - intersect)


def calc_aps(pred_bbss, gt_bbss):
    metric_fn = MetricBuilder.build_evaluation_metric('map_2d', async_mode=True, num_classes=7)
    for (pred_bbs, gt_bbs) in zip(pred_bbss, gt_bbss):
        # pred: [xmin, ymin, xmax, ymax, class_id, confidence]
        # gt:   [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt_bbs = np.append(gt_bbs, np.zeros([len(gt_bbs), 2]), axis=1) # append 2 cols
        metric_fn.add(pred_bbs.round(), gt_bbs.round())

    pascalMAP = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    pascalMAP_all = metric_fn.value(iou_thresholds=0.5)['mAP']
    cocoMAP = metric_fn.value(
        iou_thresholds=np.arange(0.5, 0.95, 0.05),
        recall_thresholds=np.arange(0., 1.01, 0.01),
        mpolicy='soft'
    )['mAP']

    print(f'VOC PASCAL mAP: {pascalMAP}')
    print(f'VOC PASCAL mAP in all points: {pascalMAP_all}')
    print(f'COCO mAP: {cocoMAP}')

def draw_bbs(imgs, bbss, color='yellow'):
    font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-L.ttf', 20)
    results = []
    for img, bbs in zip(imgs, bbss):
        draw = ImageDraw.Draw(img)
        for _, bb in enumerate(bbs):
            label = bb[4].long().item()
            bbox = bb[:4]
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=color, width=1)
            draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=font, fill=color)
        results.append(img)
    return results



class CLI(BaseMLCLI):
    class TrainArgs(BaseMLCLI.CommonArgs):
        lr:float = 0.001
        scheduler: str = Field('cosine_100', cli=('--scheduler', ))
        batch_size:int = Field(2, cli=('--batch-size', '-B'))
        num_workers:int = 4
        epoch:int = Field(50, cli=('-e', ))
        depth:str = Field('d0', cli=('-d', ))
        name:str = '{}'
        overwrite:bool = Field(False, cli=('--overwrite', '-O', ))

        @classmethod
        @validator('depth')
        def validate_depth(cls, v):
            if v not in SIZE_BY_DEPTH.keys():
                raise ValueError('Invalid depth:', v)
            return v

    def run_train(self, a):
        config = EffDetTrainerConfig(
            lr=a.lr,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            depth=a.depth,
            scheduler=a.scheduler,
        )

        dss = [
            XRBBDataset(mode='effdet', target=target, size=config.size())
            for target in ['train', 'test']
        ]

        name = a.name.format(config.depth)

        trainer = EffDetTrainer(
            config=config,
            out_dir=f'out/detection/{name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name='detection',
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)


    class PredArgs(BaseMLCLI.CommonArgs):
        checkpoint:str = Field(..., cli=('--checkpoint', '-c' ))
        src:str = Field(..., cli=('--src', '-s' ))

    def run_predict(self):
        return
        # checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        # predictor = self.create_predictor(P=EffDetPredictor, checkpoint=checkpoint)
        # paths = get_image_paths_from_dir_or_file(self.a.src)
        # images = [Image.open(p) for p in paths]
        # bbss = predictor.start(images=images)
        # results = self.draw_bbs(images, bbss)
        # dest_dir = os.path.join('out', checkpoint.name, 'predict')
        # os.makedirs(dest_dir, exist_ok=True)
        # for result, path in zip(results, paths):
        #     name = os.path.splitext(os.path.basename(path))[0]
        #     result.save(os.path.join(dest_dir, f'{name}.jpg'))


    class DsArgs(BaseMLCLI.CommonArgs):
        checkpoint:str = Field(..., cli=('--checkpoint', '-c' ))
        target:str = Field(..., cli=('--target', '-t' ), choices=['train', 'test'])

    def run_ds(self):
        return
        # checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        # predictor = self.create_predictor(P=EffDetPredictor, checkpoint=checkpoint)

        # ds = XRBBDataset(mode='effdet', target=self.a.target, size=predictor.image_size)
        # images = [i.image for i in ds.items]

        # pred_bbss = predictor.start(images=images)
        # gt_bbss = torch.stack([torch.from_numpy(i.bb.values) for i in ds.items])

        # results = self.draw_bbs(images, gt_bbss, 'green')
        # results = self.draw_bbs(results, pred_bbss, 'red')

        # dest_dir = os.path.join('out', checkpoint.name, 'dataset')
        # os.makedirs(dest_dir, exist_ok=True)
        # for result, item in zip(results, ds.items):
        #     result.save(os.path.join(dest_dir, f'{item.name}.jpg'))

    def arg_map(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])
        parser.add_argument('--length', '-l', type=int, default=-1)

    def run_map(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=EffDetPredictor, checkpoint=checkpoint)

        ds = XRBBDataset(mode='effdet', target=self.a.target, size=predictor.image_size)

        images = [i.image for i in ds.items]
        items = ds.items

        images = images[:self.a.length]
        items = items[:self.a.length]

        pred_bbss = predictor.predict_images(images=images)
        pred_bbss = [p.numpy() for p in pred_bbss]
        gt_bbss = [i.bb.values for i in items]

        calc_aps(pred_bbss, gt_bbss)

    def arg_crop(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--length', '-l', type=int, default=-1)

    def run_crop(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=EffDetPredictor, checkpoint=checkpoint)
        ds = XRBBDataset(mode='effdet', target='all', size=predictor.image_size)

        images = images=[i.image for i in ds.items]
        pred_bbss = predictor.predict_images(images)
        pred_bbss = [p.numpy() for p in pred_bbss]

        for image, pred_bbs in zip(images, pred_bbss):
            print(pred_bbs)
            break



if __name__ == '__main__':
    cli = CLI()
    cli.run()
