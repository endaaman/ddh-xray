import os
import re
from glob import glob
import math

import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import Field
from sklearn import metrics as skmetrics
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.scheduler.cosine_lr import CosineLRScheduler

import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from endaaman.ml import BaseMLCLI, BaseTrainer, BaseTrainerConfig, BaseDLArgs, Checkpoint, roc_auc_ci
from endaaman.ml.metrics import BaseMetrics

from common import cols_clinical, cols_measure, col_target, load_data
from models import TimmModelWithFeatures, TimmModel, LinearModel
from datasets import XRDataset, XRROIDataset, FeatureDataset, IMAGE_MEAN, IMAGE_STD

J = os.path.join



class ROCMetrics(BaseMetrics):
    def calc(self, preds, gts):
        if len(preds) < 10:
            return None
        preds = preds.detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        preds_bool = preds > thresholds[youden_index]
        acc = skmetrics.accuracy_score(preds_bool, gts)
        return auc, acc, tpr[youden_index], -fpr[youden_index]+1

def visualize_roc(trainer:BaseTrainer, ax, train_preds, train_gts, val_preds, val_gts):
    train_preds, train_gts, val_preds, val_gts = [
        v.detach().cpu().numpy().flatten() for v in (train_preds, train_gts, val_preds, val_gts)
    ]

    for t, preds, gts in (('train', train_preds, train_gts), ('val', val_preds, val_gts)):
        fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        lower, upper = roc_auc_ci(gts, preds)
        ax.plot(fpr, tpr, label=f'{t} AUC:{auc:.3f}({lower:.3f}-{upper:.3f})')
        if t == 'train':
            youden_index = np.argmax(tpr - fpr)
            threshold = thresholds[youden_index]

    ax.set_title(f'ROC (t={threshold:.2f})')
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.legend(loc='lower right')



class CommonTrainer(BaseTrainer):
    def get_metrics(self):
        return {
            'auc_acc_recall_spec': ROCMetrics(),
        }

    def get_visualizers(self):
        return {
            'roc': visualize_roc,
        }


class ImageTrainerConfig(BaseTrainerConfig):
    model_name: str
    with_features: bool = False
    size: int
    crop_size: int = -1
    scheduler: str
    with_features: bool
    normalize_image: bool
    normalize_features: bool

class ImageTrainer(CommonTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = TimmModelWithFeatures(
            name=self.config.model_name,
            with_features=self.config.with_features)
        return model

    def create_scheduler(self):
        if re.match(r'^cosine.*', self.config.scheduler):
            return CosineAnnealingLR(self.optimizer, T_max=50, eta_min=self.config.lr/10)
        if self.config.scheduler == 'static':
            return None
        raise RuntimeError('Invalid')

    def eval(self, inputs, gts):
        if self.config.with_features:
            inputs, features = inputs
            features = features.to(self.device)
        else:
            features = None
        inputs = inputs.to(self.device)
        outputs = self.model(inputs, features)
        loss = self.criterion(outputs, gts.to(self.device))
        return loss, outputs


class FeatureTrainerConfig(BaseTrainerConfig):
    num_features: int
    normalize_features: bool

class FeatureTrainer(CommonTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = LinearModel(num_features=self.config.num_features, num_classes=1)
        return model

    def eval(self, inputs, gts):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, gts.to(self.device))
        return loss, outputs


class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class TrainArgs(CommonArgs):
        batch_size:int = Field(2, cli=('--batch-size', '-B'))
        epoch:int = 20
        raw_features = Field(False, cli=('--raw-features', ))
        name:str = '{}'

    class ImageArgs(TrainArgs):
        lr:float = 0.0001
        fold: int = Field(None, cli=('--fold', '-f', ))
        model_name:str = Field('efficientnet_b0', cli=('--model', '-m'))
        source:str = Field('full', cli=('--source', '-s'), regex='^full|roi$')
        size:int = 512
        crop_size:int = Field(-1, cli=('--crop', ))
        raw_image = Field(False, cli=('--raw-image', ))
        mode: str = Field(False, cli=('--mode', ), regex='^image|integrated|additional$')
        scheduler:str = 'static'
        exp: str = 'classification'

    def run_image(self, a:ImageArgs):
        match a.source:
            case 'full':
                DS = XRDataset
            case 'roi':
                DS = XRROIDataset
            case _:
                raise RuntimeError('Invalid source:', a.source)

        print('Dataset type:', DS)

        basedir = 'data/folds/bag' if a.fold is None else f'data/folds6/fold{a.fold}'
        dss = [DS(
            target=t,
            basedir=J(basedir, t),
            size=a.size,
            crop_size=a.crop_size,
            num_features=8 if a.mode == 'integrated' else 0,
            normalize_image=not a.raw_image,
            normalize_features=not a.raw_features,
        ) for t in ['train', 'test']]

        config = ImageTrainerConfig(
            seed=a.seed,
            model_name=a.model_name,
            with_features=a.mode == 'integrated',
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            size=a.size,
            crop_size=a.crop_size,
            scheduler=a.scheduler,
            normalize_image=not a.raw_image,
            normalize_features=not a.raw_features,
        )
        name = a.name.format(a.model_name)
        subname = a.mode
        trainer = ImageTrainer(
            config=config,
            out_dir=f'out/{a.exp}/{subname}/{name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name=a.exp,
            main_metrics='auc',
            overwrite=a.overwrite,
        )

        # if a.mode == 'integrated':
        #     print('load weight for integrated')
        #     chp:Checkpoint = torch.load(f'out/{a.exp}/image/{name}/checkpoint_last.pt')
        #     trainer.model.load_state_dict(chp.model_state)
        #     for param in trainer.model.base.parameters():
        #         param.requires_grad = False

        # elif a.mode == 'additional':
        #     print('load weight for additional')
        #     chp:Checkpoint = torch.load(f'out/{a.exp}/image/{name}/checkpoint_last.pt')
        #     trainer.model.load_state_dict(chp.model_state)
        #     trainer.model.fc_base.reset_parameters()
        #     for param in trainer.model.base.parameters():
        #         param.requires_grad = False

        trainer.start(a.epoch)


    class FeatureArgs(TrainArgs):
        lr:float = 0.0001
        model_name:str = Field('linear', cli=('--model', '-m'))

    def run_feature(self, a:FeatureArgs):
        dss = [FeatureDataset(
            num_features=self.a.num_features,
            target=t,
            normalize_features=not a.raw_features,
        ) for t in ['train', 'test']]

        config = FeatureTrainerConfig(
            seed=a.seed,
            num_features=a.num_features,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            normalize_features=not a.raw_features,
        )
        name = a.name.format(a.model_name)
        trainer = FeatureTrainer(
            config=config,
            out_dir=f'out/classification/feature_{a.num_features}/{name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name='classification',
            main_metrics='auc',
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


    class CamArgs(CommonArgs):
        experiment_dir: str = Field(..., cli=('--exp-dir', '-e'))
        src: str
        gt: int

    def run_cam(self, a):
        checkpoint:Checkpoint = torch.load(J(a.experiment_dir, 'checkpoint_best.pt'))
        __rest_path, model_name = os.path.split(a.experiment_dir)

        model_name = re.match(r'^(.*)_fold.*$', model_name)[1]

        model = TimmModelWithFeatures(name=model_name, with_features=False)
        model.load_state_dict(checkpoint.model_state)

        image = Image.open(a.src).resize((512, 512))
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])(image.convert('L'))[None, ...]
        # result = model(t, features=None)
        gradcam = CAM.GradCAM(
            model=model,
            target_layers=[model.base.conv_head],
            use_cuda=False)
        targets = [BinaryClassifierOutputTarget(a.gt)]
        mask = gradcam(input_tensor=t, targets=targets)[0]
        visualization = show_cam_on_image(np.array(image)/255, mask, use_rgb=True)
        plt.imshow(visualization)
        plt.show()

    class CamFoldArgs(CommonArgs):
        # experiment_dir: str = Field(..., cli=('--exp-dir', '-e'))
        fold: int

    def run_cam_fold(self, a):
        df = load_data(0, True, a.seed)['all']
        experiment_dir = J('out', 'classification_effnet', 'image', f'tf_efficientnet_b0_fold{a.fold}')
        checkpoint:Checkpoint = torch.load(J(experiment_dir, 'checkpoint_best.pt'))
        __rest_path, model_name = os.path.split(experiment_dir)

        model_name = re.match(r'^(.*)_fold.*$', model_name)[1]
        model = TimmModelWithFeatures(name=model_name, with_features=False)
        model.load_state_dict(checkpoint.model_state)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

        gradcam = CAM.GradCAM(
            model=model,
            target_layers=[model.base.conv_head],
            use_cuda=False)

        dest_dir = f'out/cams/fold{a.fold}'
        os.makedirs(dest_dir, exist_ok=True)

        image_dir = f'data/folds6/fold{a.fold}/test/images'
        for p in tqdm(sorted(glob(J(image_dir, '*.jpg')))):
            id = os.path.splitext(os.path.basename(p))[0]
            image = Image.open(f'data/images/{id}.jpg').resize((512, 512))
            gt = df[df.index == id].iloc[0]['treatment'] > 0.5
            t = transform(image.convert('L'))[None, ...]
            mask = gradcam(input_tensor=t, targets=[BinaryClassifierOutputTarget(gt)])[0]

            mask_img = Image.fromarray(mask*255).convert('L')
            vis_arr = show_cam_on_image(np.array(image)/255, mask, use_rgb=True)
            vis_img = Image.fromarray(vis_arr)

            label = 'pos' if gt else 'neg'
            mask_img.save(J(dest_dir, f'{id}_mask_{label}.png'))
            vis_img.save(J(dest_dir, f'{id}_vis_{label}.png'))


            # plt.imshow(visualization)
            # plt.show()

if __name__ == '__main__':
    cli = CLI()
    cli.run()
