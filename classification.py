import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics as skmetrics
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.scheduler.cosine_lr import CosineLRScheduler

from endaaman.ml import BaseMLCLI, BaseTrainer, BaseTrainerConfig, Field, BaseDLArgs, Checkpoint
from endaaman.metrics import BaseMetrics

from common import cols_clinical, cols_measure, col_target
from models import TimmModelWithFeatures, TimmModel, LinearModel
from datasets import XRDataset, XRROIDataset, FeatureDataset

J = os.path.join


class ROCMetrics(BaseMetrics):
    def calc(self, preds, gts):
        if len(preds) < 10:
            return None
        preds = preds.detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        fpr, tpr, __thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        return auc, tpr[youden_index], -fpr[youden_index]+1

def visualize_roc(trainer:BaseTrainer, ax, train_preds, train_gts, val_preds, val_gts):
    train_preds, train_gts, val_preds, val_gts = [
        v.detach().cpu().numpy() for v in (train_preds, train_gts, val_preds, val_gts)
    ]

    for t, preds, gts in (('train', train_preds, train_gts), ('val', val_preds, val_gts)):
        fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{t} AUC:{auc:.3f}')
        if t == 'train':
            youden_index = np.argmax(tpr - fpr)
            threshold = thresholds[youden_index]
    ax.set_title(f'ROC (t={threshold:.2f})')
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.legend(loc='lower right')


def dump_preds_gts(trainer:BaseTrainer, train_preds, train_gts, val_preds, val_gts):
    train_preds, train_gts, val_preds, val_gts = [
        v.detach().cpu().numpy() for v in (train_preds, train_gts, val_preds, val_gts)
    ]

    if not trainer.is_achieved_best():
        break
    J(trainer.out_dir, 'preds.csv')
    print(val_preds.shape)
    print(val_gts.shape)


class ImageTrainerConfig(BaseTrainerConfig):
    model_name: str
    num_features: int
    size: int

class ImageTrainer(BaseTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = TimmModelWithFeatures(
            name=self.config.model_name,
            num_features=self.config.num_features)
        return model

    def eval(self, inputs, gts):
        if self.config.num_features > 0:
            inputs, features = inputs
            features = features.to(self.device)
        else:
            features = None
        inputs = inputs.to(self.device)
        outputs = self.model(inputs, features)
        loss = self.criterion(outputs, gts.to(self.device))
        return loss, outputs

    def get_metrics(self):
        return {
            'auc_recall_spec': ROCMetrics(),
        }

    def get_visualizers(self):
        return {
            'roc': visualize_roc,
        }

    def get_hooks(self):
        return {
            'dump': dump_preds_gts,
        }



class FeatureTrainerConfig(BaseTrainerConfig):
    num_features: int

class FeatureTrainer(BaseTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = LinearModel(num_features=self.config.num_features, num_classes=1)
        return model

    def eval(self, inputs, gts):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, gts.to(self.device))
        return loss, outputs

    def get_metrics(self):
        return {
            'auc_recall_spec': ROCMetrics(),
        }

    def get_visualizers(self):
        return {
            'roc': visualize_roc,
        }

    def get_hooks(self):
        return {
            'dump': dump_preds_gts,
        }


class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class ImageArgs(CommonArgs):
        lr:float = 0.0001
        model_name:str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        num_features:int = Field(0, cli=('--num-features', '-F'))
        batch_size:int = Field(8, cli=('--batch-size', '-B'))
        source:str = Field('full', regex='^full|roi$')
        size:int = 512
        suffix:str = ''
        epoch:int = 20

    def run_image(self, a:ImageArgs):
        match a.source:
            case 'full':
                DS = XRDataset
            case 'roi':
                DS = XRROIDataset
            case _:
                raise RuntimeError('Invalid source:', a.source)

        print('Dataset type:', DS)
        dss = [DS(
            size=a.size,
            num_features=self.a.num_features,
            target=t,
        ) for t in ['train', 'test']]

        config = ImageTrainerConfig(
            model_name=a.model_name,
            num_features=a.num_features,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            size=a.size,
        )
        name = f'{a.model_name}_{a.suffix}' if a.suffix else a.model_name
        trainer = ImageTrainer(
            config=config,
            out_dir=f'out/classification/{a.source}_{a.num_features}/{name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name='classification',
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)


    class FeatureArgs(BaseDLArgs):
        model_name:str = Field('linear', cli=('--model', '-m'))
        lr:float = 0.0001
        num_features:int = Field(8, cli=('--num-features', '-F'))
        batch_size:int = Field(8, cli=('--batch-size', '-B'))
        suffix:str = ''
        epoch:int = 20

    def run_feature(self, a:FeatureArgs):
        dss = [FeatureDataset(
            num_features=self.a.num_features,
            target=t,
        ) for t in ['train', 'test']]

        config = FeatureTrainerConfig(
            num_features=a.num_features,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
        )
        name = f'{a.model_name}_{a.suffix}' if a.suffix else a.model_name
        trainer = FeatureTrainer(
            config=config,
            out_dir=f'out/classification/feature_{a.num_features}/{name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name='classification',
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)

    class PredictArgs(CommonArgs):
        experiment_dir: str = Field(..., cli=('--exp-dir', '-e'))

    # def run_predict(self, a:PredictArgs):
    #     checkpoint:Checkpoint = torch.load(J(a.experiment_dir, 'checkpoint_best.pt'))
    #     print(checkpoint.config)
    #     rest_path, model_name = os.path.split(a.experiment_dir)

    #     # remove trailing suffix number
    #     model_name = re.sub('_\d*$', '', model_name, 1)

    #     rest_path, mode = os.path.split(rest_path)

    #     source, num_features = mode.split('_')
    #     num_features = int(num_features)

    #     match model_name:
    #         case 'linear':
    #             model = LinearModel(num_features, 1)
    #             config = FeatureTrainerConfig(**checkpoint.config)
    #         case _:
    #             model = TimmModelWithFeatures(model_name, num_features, 1)
    #             config = ImageTrainerConfig(**checkpoint.config)

    #     ds = XRDataset(size=config.size, num_features=num_features, target='test')

    # def arg_ds(self, parser):
    #     parser.add_argument('--checkpoint', '-c', required=True)
    #     parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    # def run_ds(self):
    #     checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
    #     predictor = self.create_predictor(P=ClsPredictor, checkpoint=checkpoint)

    #     ds = XRDataset(target=self.a.target, with_features=predictor.with_features, aug_mode='test')
    #     loader = DataLoader(dataset=ds, batch_size=self.a.batch_size, num_workers=1)

    #     results = predictor.predict(loader=loader)

    #     pred_y = np.array(results)
    #     true_y = np.array([i.treatment for i in ds.items])
    #     print(metrics.roc_auc_score(true_y, pred_y))

    #     print('done')


if __name__ == '__main__':
    cli = CLI()
    cli.run()
