import os
import re
from glob import glob
from typing import NamedTuple

from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms

from effdet import DetBenchPredict
from endaaman import get_paths_from_dir_or_file
from endaaman.torch import TorchCommander

from models import create_det_model
from datasets import XRBBDataset, XRROIDataset, LABEL_TO_STR, IMAGE_MEAN, IMAGE_STD


COLOR = 'yellow'

def select_best_bbs(bbs):
    '''
    bbs: [[x0, y0, x1, y1, cls]]
    '''
    best_bbs = []
    missing = []
    for label, text in LABEL_TO_STR.items():
        m = bbs[bbs[:, 4].long() == label]
        if len(m) > 0:
            # select first bb
            best_bbs.append(m[0])
        else:
            missing.append(text)
    return torch.stack(best_bbs), ' '.join(missing)


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--batch-size', '-b', type=int, default=8)

    def pre_common(self):
        self.checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        # self.checkpoint = torch.load(self.args.checkpoint)

        self.model, self.image_size = create_det_model(self.checkpoint.name)
        self.model.load_state_dict(self.checkpoint.model_state)

        self.model.eval()
        self.bench = DetBenchPredict(self.model).to(self.device)

        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-L.ttf', 20)


    def detect_rois(self, imgs, image_size):
        scales = [torch.tensor(i.size)/image_size for i in imgs]
        imgs = [i.resize((image_size, image_size)) for i in imgs]

        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([IMAGE_MEAN]*3, [IMAGE_STD]*3),
        ])

        bs = self.args.batch_size
        bbss = []
        start = 0
        idx = 0
        t = tqdm(range(0, len(imgs), bs))
        for start in t:
            batch = imgs[start:start + bs]

            inputs = torch.stack([transform_image(i) for i in batch]).to(self.device)

            with torch.no_grad():
                batch_bbss = self.bench(inputs).detach().cpu()

                batch_bbss[:, :, 4] = batch_bbss[:, :, 5]
                for bbs in batch_bbss:
                    bbs, missing = select_best_bbs(bbs)
                    if len(missing) > 0:
                        print(f'[{idx}] missing: {missing}')
                    # bbs: [[x0, y0, x1, y1, cls, index]]
                    bbs[:, 5] = idx
                    # (w, h) -> (w, w, h, h)
                    scale = scales[idx].repeat_interleave(2)
                    bbs[:, :4] *= scale
                    bbss.append(bbs)
                    idx += 1

            t.set_description(f'{start} ~ {start + bs} / {len(imgs)}')
            t.refresh()

        return bbss

    def draw_bbs(self, imgs, bbss):
        results = []

        for img, bbs in zip(imgs, bbss):
            draw = ImageDraw.Draw(img)
            for _, bb in enumerate(bbs):
                label = bb[4].item()
                bbox = bb[:4]
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)
                draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=self.font, fill='yellow')
            results.append(img)
        return results

    def arg_xr(self, parser):
        parser.add_argument('--src', '-s', required=True)

    def run_xr(self):
        paths = get_paths_from_dir_or_file(self.a.src)
        images = [Image(p) for p in paths]

        bbss = self.detect_rois(images, self.image_size)
        results = self.draw_bbs(images, bbss)

        dest_dir = os.path.join('out', self.checkpoint.name, 'predict')
        os.makedirs(dest_dir, exist_ok=True)
        for result, path in zip(results, paths):
            name = os.path.splitext(os.path.basename(path))[0]
            result.save(os.path.join(dest_dir, f'{name}.jpg'))

        print('done')

    def arg_roi(self, parser):
        parser.add_argument('--src', '-s', required=True)

    def run_roi(self):
        paths = get_paths_from_dir_or_file(self.a.src)
        images = [Image(p) for p in paths]

        bbss = self.detect_rois(images, self.image_size)
        results = self.draw_bbs(images, bbss)

        dest_dir = os.path.join('out', self.checkpoint.name, 'predict')
        os.makedirs(dest_dir, exist_ok=True)
        for result, path in zip(results, paths):
            name = os.path.splitext(os.path.basename(path))[0]
            result.save(os.path.join(dest_dir, f'{name}.jpg'))

        print('done')

    def arg_map(self, parser):
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    def run_map(self):
        ds = XRBBDataset(mode='effdet', target=self.a.target, size=self.image_size)

        images = [i.image for i in ds.items]
        bbss = self.detect_rois(images, self.image_size)

        for (bbs, item) in zip(bbss, ds.items):
            print(bbs)
            print(item.bb)
            self.bbs = bbs
            self.item = item
            break



if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
