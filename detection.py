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
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from effdet import DetBenchTrain, DetBenchPredict
from ptflops import get_model_complexity_info
from timm.scheduler import CosineLRScheduler
from mean_average_precision import MetricBuilder

from endaaman import get_image_paths_from_dir_or_file
from endaaman.torch import TorchCommander, Trainer, Predictor

from datasets import XRBBDataset, LABEL_TO_STR, IMAGE_STD, IMAGE_MEAN
from utils import get_state_dict
from models import create_det_model, yolor_loss, SIZE_BY_DEPTH
from models.ssd import MultiBoxLoss



class EffDetTrainer(Trainer):
    def prepare(self, **kwargs):
        self.bench = DetBenchTrain(self.model).to(self.device)

    def create_model(self):
        return create_det_model(self.model_name)

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
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])

    def create_model(self):
        model = create_det_model(self.checkpoint.model_name)
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
        # bbs[:, 5] = idx
        return bbs

    def predict_images(self, images):
        scales = [torch.tensor(i.size)/self.image_size for i in images]
        images = [i.resize((self.image_size, self.image_size)) for i in images]

        bbss = super().predict_images(images)
        for bbs, scale in zip(bbss, scales):
            scale = scale.repeat_interleave(2)
            bbs[:, :4] *= scale
        return bbss


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

    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
    print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 0.95, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")



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
        loaders = self.create_loaders(mode='effdet', size=SIZE_BY_DEPTH[self.a.depth])

        trainer = self.create_trainer(
            T=EffDetTrainer,
            model_name=name,
            loaders=loaders,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


    def draw_bbs(self, imgs, bbss, color='yellow'):
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

    def arg_predict(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--src', '-s', required=True)

    def run_predict(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=EffDetPredictor, checkpoint=checkpoint)

        paths = get_image_paths_from_dir_or_file(self.a.src)
        images = [Image.open(p) for p in paths]

        bbss = predictor.start(images=images)
        results = self.draw_bbs(images, bbss)

        dest_dir = os.path.join('out', checkpoint.name, 'predict')
        os.makedirs(dest_dir, exist_ok=True)
        for result, path in zip(results, paths):
            name = os.path.splitext(os.path.basename(path))[0]
            result.save(os.path.join(dest_dir, f'{name}.jpg'))

        print('done')


    def arg_ds(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    def run_ds(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=EffDetPredictor, checkpoint=checkpoint)

        ds = XRBBDataset(mode='effdet', target=self.a.target, size=predictor.image_size)
        images = [i.image for i in ds.items]

        pred_bbss = predictor.start(images=images)
        gt_bbss = torch.stack([torch.from_numpy(i.bb.values) for i in ds.items])

        results = self.draw_bbs(images, gt_bbss, 'green')
        results = self.draw_bbs(results, pred_bbss, 'red')

        dest_dir = os.path.join('out', checkpoint.name, 'dataset')
        os.makedirs(dest_dir, exist_ok=True)
        for result, item in zip(results, ds.items):
            result.save(os.path.join(dest_dir, f'{item.name}.jpg'))

        print('done')

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
    cmd = CMD({
        'epoch': 100,
        'lr': 0.01,
        'batch_size': 8,
        'save_period': 25,
    })
    cmd.run()
