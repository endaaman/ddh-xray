import os
import json

from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from augmentation import Augmentation, PlaneAugmentation
from datasets import XrayDataset
from endaaman import Trainer


class MyTrainer(Trainer):
    def arg_common(self, parser):
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-n', '--num-workers', type=int, default=os.cpu_count()//2)
        parser.add_argument('-b', '--batch-size', type=int, default=48)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('--network', default='d0', type=str, help='efficientdet-[d0, d1, ..]')

    def save_model(self, model, epoch):
        state = {
            'epoch': epoch,
            'parser': self.args,
            'state_dict': model.get_state_dict(),
        }
        weights_path = f'weights/{epoch}.pth'
        torch.save(state, weights_path)
        return weights_path

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
            aug = PlaneAugmentation()
        else:
            aug = Augmentation()
        train_dataset.set_augmentaion(aug)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        return train_loader, None

    def run_train(self):
        train_loader, __test_loader = self.create_loaders()

        network = f'efficientdet-{self.args.network}'
        model = EfficientDet(
            num_classes=1,
            network=network,
            W_bifpn=EFFDET_PARAMS[network]['W_bifpn'],
            D_bifpn=EFFDET_PARAMS[network]['D_bifpn'],
            D_class=EFFDET_PARAMS[network]['D_class'])
        model = model.to(self.device)

        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        model.train()
        model.freeze_bn()

        print('Starting training')
        for epoch in range(1, self.args.epoch + 1):
            model.train()
            t = tqdm(train_loader)
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
                break

            #* validate

            #* draw fig
            if epoch % 2 == 0:
                pass

            #* save weights
            if epoch % 1 == 0:
                self.save_model(model, epoch)

            scheduler.step()


        print('anchors shape:', anchors.shape)
        print('classification shape:', classification.shape)
        print('regression shape:', regression.shape)
        classification_loss, regression_loss = criterion(classification, regression, anchors, bb, self.device)
        print(classification_loss)
        print(regression_loss)


MyTrainer().run()
