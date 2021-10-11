import os
import json

from PIL import Image, ImageDraw, ImageFont
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
        dest_dir = self.args.output_dir
        if not os.path.isdir(dest_dir):
            print(f'{output_base} is not directory')
            exit(1)

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
        results = results[0].type(torch.long)
        bboxes = []
        bbox_scores = []
        label_to_str = {
            1: 'left top',
            2: 'left out',
            3: 'left in',
            4: 'right top',
            5: 'right out',
            6: 'right in',
        }
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)
        boxes = []
        for i in label_to_str.keys():
            m = results[results[:, 5] == i]
            if len(m) > 0:
                boxes.append(m[0])
        for i, result in enumerate(boxes):
            bbox = result[:4]
            label = result[5].item()
            draw.text((bbox[0], bbox[1]), label_to_str[label], font=font, fill='yellow')
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)


        # model_id = state['args'].network + '_e' + str(state['epoch'])
        # dest_dir = os.path.join(self.args.dest, model_id)
        file_name = os.path.basename(os.path.abspath(self.args.input))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_name)
        img.save(dest_path)
        print(f'saved {dest_path}')
        os.system(f'xdg-open {dest_path}')


Predictor().run()
