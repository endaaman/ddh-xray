import os
import json

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from effdet import EfficientDet, DetBenchPredict, get_efficientdet_config
from effdet.efficientdet import HeadNet

from augmentation import ResizeAugmentation
# from datasets import XrayDataset
from utils import pil_to_tensor

from endaaman import TorchCommander


class Predictor(TorchCommander):
    def create_model(self, network):
        cfg = get_efficientdet_config(f'tf_efficientdet_{network}')
        cfg.num_classes = 6
        model = EfficientDet(cfg, pretrained_backbone=True)
        # model.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
        return model

    def run_test(self):
        model = self.create_model('d0')
        bench = DetBenchPredict(model)
        images = torch.randn(2, 3, 512, 512)
        results = bench(images)

        preds_boxes = preds_boxes.type(torch.LongTensor)
        # print(preds_boxes)
        # print(results.shape)

    def arg_single(self, parser):
        parser.add_argument('-w', '--weights', type=str, required=True)
        parser.add_argument('-i', '--input', type=str, required=True)
        parser.add_argument('-o', '--output-dir', type=str, default='tmp')

    def run_single(self):
        state = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        model = self.create_model(state['args'].network)
        model.load_state_dict(state['state_dict'])
        bench = DetBenchPredict(model).to(self.device)

        img_size = 512
        img = Image.open(self.args.input)
        img = img.resize((img_size, img_size))
        img_tensor = pil_to_tensor(img)
        # img_tensor = img_tensor.permute([0, 2, 1])
        img_tensor = img_tensor[None, :, :, :]

        results = bench(img_tensor.to(self.device))
        results = results.type(torch.long)
        print('results', results.shape)
        return

        bboxes = []
        bbox_scores = []
        draw = ImageDraw.Draw(img)
        for result in results[0]:
            bbox = result[:4]
            print(bbox)
            label = result[5]
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)

        file_name = os.path.basename(os.path.abspath(self.args.src))
        model_id = state['args'].network + '_e' + str(state['epoch'])
        dest_dir = os.path.join(self.args.dest, model_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_name)
        img.save(dest_path)
        print(f'saved {dest_path}')
        os.system(f'xdg-open {dest_path}')


Predictor().run()
