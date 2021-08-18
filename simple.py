import os
import json

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from augmentation import Augmentation, ResizeAugmentation, CropAugmentation
from datasets import XrayDataset, ROIDataset
from utils import get_state_dict
from models import VGG
from metrics import calc_acc, calc_recall, calc_spec
from endaaman import Trainer


class MyTrainer(Trainer):
    def run_check(self, args):
        model = VGG(name='vgg16_bn', num_classes=1, pretrained=True)
        t = torch.randn(3, 3, 224, 224)
        print(model(t).size())

    def arg_common(self, parser):
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-b', '--batch-size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)

    def create_loaders(self):
        train_dataset = ROIDataset()
        transform_x = transforms.Compose([
            transforms.ToTensor(),
            train_dataset.get_normalizer(),
            lambda x: x.permute([0, 2, 1]),
        ])
        transform_y = transforms.Compose([
            lambda y: torch.tensor(y, dtype=torch.float32),
            lambda y: y.unsqueeze(0),
        ])
        train_dataset.set_transforms(transform_x, transform_y)
        # if not self.args.no_aug:
        #     aug = Augmentation(tile_size=tile_size)
        # train_dataset.set_augmentaion(aug)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )
        return train_loader, None

    def arg_train(self, parser):
        parser.add_argument('--model', default='vgg16_bn')

    def save_state(self, model, epoch):
        state = {
            'epoch': epoch,
            'args': self.args,
            'state_dict': get_state_dict(model),
        }
        weights_dir = f'weights/{self.args.model}'
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f'{epoch}.pth')
        torch.save(state, weights_path)
        return weights_path

    def train_epoch(self, model, loader, criterion, optimizer, get_message):
        metrics = {
            'loss': [],
            'acc': [],
            'recall': [],
            'spec': [],
        }
        t = tqdm(loader, leave=False)
        for (inputs, labels) in t:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_metrics = {
                'loss': float(loss.item()),
                'acc': calc_acc(outputs, labels),
                'recall': calc_recall(outputs, labels),
                'spec': calc_spec(outputs, labels),
            }
            message = ' '.join([f'{k}:{v:.3f}' for k, v in iter_metrics.items()])
            t.set_description(get_message(message))
            t.refresh()
            for k, v in iter_metrics.items():
                metrics[k].append(v)

        return {k: np.mean(v) for k, v in metrics.items()}

    def run_train(self, args):
        model = VGG(name=args.model, num_classes=1, pretrained=True).to(self.device)
        train_loader, __test_loader = self.create_loaders()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        criterion = nn.BCELoss()
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95 ** x)

        print('Starting training')
        for epoch in range(1, args.epoch + 1):
            header = f'[{epoch}/{args.epoch}] '

            # lr = scheduler.get_last_lr()[0]
            lr = optimizer.param_groups[0]['lr']
            print(f'{header}Starting lr={lr:.7f}')

            train_metrics = self.train_epoch(
                model, train_loader, criterion, optimizer,
                lambda m: f'{header}{m}'
            )
            train_message = ' '.join([f'{k}:{v:.4f}' for k, v in train_metrics.items()])
            print(f'{header}Train: {train_message}')

            #* validate

            #* draw fig
            if epoch % 2 == 0:
                pass

            #* save weights
            if epoch % 20 == 0:
                weights_path = self.save_state(model, epoch)
                print(f'{header}Saved "{weights_path}"')

            scheduler.step(train_metrics['loss'])
            # scheduler.step()
            print()


MyTrainer().run()
