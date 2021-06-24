import os
import json

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from augmentation import Augmentation, ResizeAugmentation
from datasets import XrayDataset
from endaaman import Trainer


class MyTrainer(Trainer):
    def run_check(self):
        model = self.create_model(self.args.network)

        images = torch.randn(2, 3, 640, 640)
        labels = torch.ones(2, 2, 5, dtype=torch.float)
        print(images.shape)
        print(labels.shape)
        criterion = FocalLoss()
        classification, regression, anchors = model(images)
        print('anchors shape:', anchors.shape)
        print('classification shape:', classification.shape)
        print('regression shape:', regression.shape)
        classification_loss, regression_loss = criterion(classification, regression, anchors, labels, 'cpu')
        print(classification_loss)
        print(regression_loss)

    def arg_common(self, parser):
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-n', '--num-workers', type=int, default=os.cpu_count()//2)
        parser.add_argument('-b', '--batch-size', type=int, default=48)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('--network', default='d0', type=str, help='efficientdet-[d0, d1, ..]')

    def save_state(self, model, epoch):
        state = {
            'epoch': epoch,
            'args': self.args,
            'state_dict': model.get_state_dict(),
        }
        weights_path = f'weights/{epoch}.pth'
        torch.save(state, weights_path)
        return weights_path

    def create_model(self, network):
        network = f'efficientdet-{network}'
        model = EfficientDet(
            num_classes=6,
            network=network,
            W_bifpn=EFFDET_PARAMS[network]['W_bifpn'],
            D_bifpn=EFFDET_PARAMS[network]['D_bifpn'],
            D_class=EFFDET_PARAMS[network]['D_class'])
        return model

    def create_loaders(self):
        train_dataset = XrayDataset()
        transform_x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*[[v] * 3 for v in train_dataset.get_mean_and_std()]),
        ])
        transform_y = transforms.Compose([
            lambda aa: [[a.id, *a.rect] for a in aa],
            lambda y: torch.tensor(y, dtype=torch.float),
        ])
        train_dataset.set_transforms(transform_x, transform_y)
        if self.args.no_aug:
            aug = ResizeAugmentation(tile_size=512)
        else:
            aug = Augmentation(tile_size=512)
        train_dataset.set_augmentaion(aug)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        return train_loader, None

    def do_train(self, model, starting_epoch):
        train_loader, __test_loader = self.create_loaders()

        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        model.train()
        model.freeze_bn()
        print('Starting training')
        for epoch in range(starting_epoch, self.args.epoch + 1):
            header = f'[{epoch}/{self.args.epoch}] '
            model.train()
            t = tqdm(train_loader)
            print(f'{header}Starting lr={optimizer.param_groups[0]["lr"]:.5f}')
            epoch_loss = []
            for (inputs, labels) in t:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                classification, regression, anchors = model(inputs)
                classification_loss, regression_loss = criterion(
                    classification, regression, anchors, labels, self.device)
                loss = classification_loss.mean() + regression_loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                metrics = {
                    'loss': loss.item(),
                }
                message = ' '.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                t.set_description(message)
                t.refresh()
                epoch_loss.append(loss.item())

            #* validate

            #* draw fig
            if epoch % 2 == 0:
                pass

            #* save weights
            if epoch % 1 == 0:
                weights_path = self.save_state(model, epoch)
                print(f'{header}Saved "{weights_path}"')
            scheduler.step(np.mean(epoch_loss))
            print(f'{header}Done.')

    def arg_start(self, parser):
        pass

    def run_start(self):
        model = self.create_model(self.args.network)
        self.do_train(model, 1)

    def arg_restore(self, parser):
        parser.add_argument('-w', '--weights', type=str, required=True)

    def pre_restore(self):
        pass

    def run_restore(self):
        pass
        # state = torch.load(self.args.weights)
        # model = self.create_model(self.args.network)
        # model.load_state_dict(torch.load(state['state_dict']))
        # self.do_train(model, state[epoch] + 1)



MyTrainer().run()
