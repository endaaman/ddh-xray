import os
import json

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
# from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from effdet.efficientdet import HeadNet
# from augmentation import Augmentation, ResizeAugmentation, CropAugmentation
from datasets import EffdetDataset
from utils import get_state_dict

from endaaman import TorchCommander


# SIZE_BY_NETWORK = {f'd{d}': 128 * (4+d) for d in range(8)}
SIZE_BY_NETWORK = {
    'd0': 128 * 4,
    'd1': 128 * 5,
    'd2': 128 * 6,
    'd3': 128 * 7,
    'd4': 128 * 8,
    'd5': 128 * 10,
    'd6': 128 * 12,
    'd7': 128 * 14,
}


class MyTrainer(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('-m', '--model', default='d0', type=str, choices=SIZE_BY_NETWORK.keys() + ['yolo'])
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-b', '--batch-size', type=int, default=48)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)

    def run_check(self):
        cfg = get_efficientdet_config(f'tf_efficientdet_{self.args.network}')
        cfg.image_size
        model = EfficientDet(cfg)
        bench = DetBenchTrain(model)
        s = SIZE_BY_NETWORK[self.args.network]
        images = torch.randn(2, 3, s, s)
        targets = {
            'bbox': torch.FloatTensor(
                [
                    [
                        [0, 0, 20, 30],
                    ]
                ]
            ),
            'cls': torch.LongTensor( [ [ 1, ],]) ,
        }
        loss = bench(images, targets)

    def save_weights(self, model, epoch):
        state = {
            'epoch': epoch,
            'args': self.args,
            'state_dict': get_state_dict(model),
        }
        weights_dir = f'weights/{self.args.network}'
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f'{epoch}.pth')
        torch.save(state, weights_path)
        return weights_path

    def create_model(self, network):
        cfg = get_efficientdet_config(f'tf_efficientdet_{network}')
        cfg.num_classes = 6
        model = EfficientDet(cfg)
        # model.class_net = HeadNet(
        #     config,
        #     num_outputs=config.num_outputs,
        # )
        return model

    def create_loaders(self, target='effdet'):
        train_dataset = EffdetDataset()
        tile_size = SIZE_BY_NETWORK.get(self.args.network, 512)
        if not self.args.no_aug:
            train_dataset.apply_augs([
                A.RandomResizedCrop(width=tile_size, height=tile_size, scale=[0.7, 1.0]),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
            ])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )
        return train_loader, None

    def train_epoch(self, bench, loader, optimizer, get_message):
        metrics = {
            'loss': [],
        }
        t = tqdm(loader, leave=False)
        for (inputs, targets) in t:
            inputs = inputs.to(self.device)
            targets['bbox'] = targets['bbox'].to(self.device)
            targets['cls'] = targets['cls'].to(self.device)
            optimizer.zero_grad()
            losses = bench(inputs, targets)
            loss = losses['loss']
            loss.backward()
            optimizer.step()
            iter_metrics = {
                'loss': float(loss.item()),
            }
            message = ' '.join([f'{k}:{v:.4f}' for k, v in iter_metrics.items()])
            t.set_description(get_message(message))
            t.refresh()
            for k, v in iter_metrics.items():
                metrics[k].append(v)

        return {k: np.mean(v) for k, v in metrics.items()}

    def arg_train(self, parser):
        pass

    def run_train(self):
        model = self.create_model(self.args.network)
        bench = DetBenchTrain(model).to(self.device)

        train_loader, _ = self.create_loaders()

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95 ** x)

        print('Starting training')
        for epoch in range(1, self.args.epoch + 1):
            header = f'[{epoch}/{self.args.epoch}] '

            # lr = scheduler.get_last_lr()[0]
            lr = optimizer.param_groups[0]['lr']
            print(f'{header}Starting lr={lr:.7f}')

            train_metrics = self.train_epoch(
                bench, train_loader, optimizer,
                lambda m: f'{header}{m}'
            )
            train_message = ' '.join([f'{k}:{v:.4f}' for k, v in train_metrics.items()])
            print(f'{header}Train: {train_message}')

            #* validate

            #* draw fig
            if epoch % 2 == 0:
                pass

            #* save weights
            if epoch % 10 == 0:
                weights_path = self.save_weights(bench.model, epoch)
                print(f'{header}Saved "{weights_path}"')

            scheduler.step(train_metrics['loss'])
            # scheduler.step()
            print()

    def run_fake_weights(self):
        model = self.create_model(self.args.network)
        weights_path = self.save_weights(model, 0)
        print(weights_path)


MyTrainer().run()
