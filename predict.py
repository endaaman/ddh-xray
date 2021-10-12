import os
import json
from glob import glob

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
from datasets import XRBBDataset
from utils import pil_to_tensor

from endaaman import TorchCommander


SIZE_BY_NETWORK= {
    'd0': 512,
    'd1': 640,
}

LABEL_TO_STR = {
    1: 'right top',
    2: 'right out',
    3: 'right in',
    4: 'left top',
    5: 'left out',
    6: 'left in',
}

class Predictor(TorchCommander):
    def create_bench(self, state):
        n = state['args'].network
        cfg = get_efficientdet_config(f'tf_efficientdet_{n}')
        cfg.num_classes = 6
        model = EfficientDet(cfg, pretrained_backbone=True)
        model.load_state_dict(state['state_dict'])
        bench = DetBenchPredict(model).to(self.device)
        bench.eval()
        return bench

    def pre_common(self):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)

    def arg_common(self, parser):
        parser.add_argument('-b', '--batch-size', type=int, default=16)

    def arg_dir(self, parser):
        parser.add_argument('-w', '--weights', type=str, required=True)
        parser.add_argument('-i', '--input-dir', type=str, required=True)
        parser.add_argument('-o', '--output-dir', type=str, default='tmp')

        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)

    def run_dir(self):
        os.makedirs(self.args.output_dir, exist_ok=True)

        state = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        bench = self.create_bench(state)

        image_size = SIZE_BY_NETWORK.get(state['args'].network, 512)
        imgs = []
        names = []
        for p in glob(os.path.join(self.args.input_dir, '*')):
            imgs.append(Image.open(p))
            names.append(os.path.basename(p))

        if len(imgs) < 1:
            print('empty')
            return
        results = self.detect_images(bench, imgs, image_size)

        for name, result in tqdm(zip(names, results)):
            p = os.path.join(self.args.output_dir, name)
            result.save(p)
        print('done')

    def detect_images(self, bench, imgs, size=512):
        imgs = [i.resize((size, size)) for i in imgs]

        bs = self.args.batch_size
        outputs = []
        start = 0
        t = tqdm(range(0, len(imgs), bs))
        for start in t:
            batch = imgs[start:start + bs]

            tt = torch.stack([pil_to_tensor(i) for i in batch]).to(self.device)
            with torch.no_grad():
                output_tensor = bench(tt)
            outputs.append(output_tensor.detach().cpu())
            t.set_description(f'{start} ~ {start + bs} / {len(imgs)}')
            t.refresh()
        outputs = torch.cat(outputs).type(torch.long)

        results = []
        print(outputs.shape)
        for img, bboxes in zip(imgs, outputs):
            best_bboxes = []
            for i in LABEL_TO_STR.keys():
                m = bboxes[bboxes[:, 5] == i]
                if len(m) > 0:
                    best_bboxes.append(m[0])
                else:
                    print('missing {LABEL_TO_STR[i]}')

            draw = ImageDraw.Draw(img)
            for i, result in enumerate(best_bboxes):
                bbox = result[:4]
                label = result[5].item()
                draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=self.font, fill='yellow')
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)
            results.append(img)
        return results


    def arg_single(self, parser):
        parser.add_argument('-w', '--weights', type=str, required=True)
        parser.add_argument('-i', '--input', type=str, required=True)
        parser.add_argument('-o', '--output-dir', type=str, default='tmp')

    def run_single(self):
        dest_dir = self.args.output_dir
        os.makedirs(dest_dir, exist_ok=True)
        state = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        bench = self.create_bench(state)

        img_size = 512
        img = Image.open(self.args.input)
        img = self.detect_images(bench, [img])[0]

        # model_id = state['args'].network + '_e' + str(state['epoch'])
        # dest_dir = os.path.join(self.args.dest, model_id)
        file_name = os.path.basename(os.path.abspath(self.args.input))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_name)
        img.save(dest_path)
        print(f'saved {dest_path}')
        os.system(f'xdg-open {dest_path}')


Predictor().run()
