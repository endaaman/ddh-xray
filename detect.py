import os
import json

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from augmentation import Augmentation, ResizeAugmentation, CropAugmentation
from datasets import XrayDataset
from utils import get_state_dict
from endaaman import TorchCommander


def label_to_tensor(label, device):
    # return torch.from_numpy(label.values).type(torch.FloatTensor)
    return {
        'bbox': torch.FloatTensor(label.values[:, :4]),
        'cls': torch.FloatTensor(label.values[:, 4]),
    }

SIZE_BY_NETWORK= {
    'd0': 512,
    'd1': 640,
}

class MyTrainer(TorchCommander):
    def run_check(self, args):
        # model = self.create_model(self.args.network)
        cfg = get_efficientdet_config(f'tf_efficientdet_{self.args.network}')
        model = EfficientDet(cfg)
        bench = DetBenchTrain(model)

        images = torch.randn(2, 3, 512, 512)
        # labels = torch.ones(2, 2, 5, dtype=torch.float)
        targets = {
            'bbox': torch.FloatTensor(
                [
                    [
                        [0, 0, 20, 30],
                        [0, 0, 20, 30],
                    ],
                    [
                        [0, 0, 20, 30],
                        [0, 0, 20, 30],
                    ]
                ]
            ),
            'cls': torch.LongTensor( [ [ 1, 2 ], [ 1, 2 ], ]) ,
        }
        print(targets['bbox'].shape)
        loss = bench(images, targets)
        # criterion = FocalLoss()
        # classification, regression, anchors = model(images)
        # print('anchors shape:', anchors.shape)
        # print('classification shape:', classification.shape)
        # print('regression shape:', regression.shape)
        # classification_loss, regression_loss = criterion(classification, regression, anchors, labels, 'cpu')
        # print(classification_loss)
        # print(regression_loss)

    def arg_common(self, parser):
        parser.add_argument('-n', '--network', default='d0', type=str, help='efficientdet-[d0, d1, ..]')
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-b', '--batch-size', type=int, default=48)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)

    def save_state(self, model, epoch):
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
        return model

    def create_loaders(self):
        train_dataset = XrayDataset()
        transform_x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*[[v] * 3 for v in train_dataset.get_mean_and_std()]),
            lambda x: x.permute([0, 2, 1]),
        ])
        transform_y = transforms.Compose([
            lambda y: label_to_tensor(y, self.device),
        ])
        train_dataset.set_transforms(transform_x, transform_y)
        tile_size = SIZE_BY_NETWORK[self.args.network]
        if self.args.no_aug:
            aug = ResizeAugmentation(tile_size=tile_size)
            # aug = CropAugmentation(tile_size=tile_size)
        else:
            aug = Augmentation(tile_size=tile_size)
        train_dataset.set_augmentaion(aug)

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

    def arg_yolo(self, parser):
        pass

    def run_yolo(self):
        model = self.create_model(self.args.network)
        bench = DetBenchTrain(model).to(self.device)

        train_loader, __test_loader = self.create_loaders()

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95 ** x)

        print('Starting training')
        for epoch in range(1, self.args.epoch + 1):
            header = f'[{epoch}/{self.args.epoch}] '

            # lr = scheduler.get_last_lr()[0]
            lr = optimizer.param_groups[0]["lr"]
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
                weights_path = self.save_state(bench.model, epoch)
                print(f'{header}Saved "{weights_path}"')

            scheduler.step(train_metrics['loss'])
            # scheduler.step()
            print()


MyTrainer().run()
