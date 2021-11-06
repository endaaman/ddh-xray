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
from models import YOLOv3, Yolor
from models.yolo_v3 import non_max_suppression, rescale_boxes
from utils import pil_to_tensor

from endaaman import TorchCommander


SIZE_BY_DEPTH = {
    'd0': 128 * 4,
    'd1': 128 * 5,
    'd2': 128 * 6,
    'd3': 128 * 7,
    'd4': 128 * 8,
    'd5': 128 * 10,
    'd6': 128 * 12,
    'd7': 128 * 14,
}

LABEL_TO_STR = {
    1: 'right top',
    2: 'right out',
    3: 'right in',
    4: 'left top',
    5: 'left out',
    6: 'left in',
}

class EffdetPredictor:
    def __init__(self, bench, model, image_size):
        self.bench = bench
        self.model = model
        self.image_size = image_size

    def get_image_size(self):
        return self.image_size

    def __call__(self, inputs):
        pred_clss, pred_boxes = self.model(inputs)
        outputs = self.bench(inputs).detach().cpu().type(torch.long)
        outputs[:, :, 4] = outputs[:, :, 5]
        return [o for o in outputs]

class YOLOPredictor:
    def __init__(self, model, conf_thres, nms_thres):
        self.model = model
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def get_image_size(self):
        return 512

    def __call__(self, inputs):
        tt = self.model.predict(inputs, self.conf_thres, self.nms_thres)
        outputs = []
        for t in tt:
            if t is None:
                outputs.append(torch.tensor([]))
            else:
                t = t.type(torch.long)
                t[:, 4] = t[:, 6]
                outputs.append(t[:, :5])
        return outputs


class YolorPredictor:
    def __init__(self, model, conf_thres, nms_thres):
        self.model = model
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def get_image_size(self):
        return 512

    def __call__(self, inputs):
        tt = self.model.predict(inputs, conf_thres=self.conf_thres, iou_thres=0.213, agnostic_nms=False)
        outputs = []
        for det in tt:
            print(det)
            outputs.append(det)
            # import pdb; pdb.set_trace()
            # if t is None:
            #     outputs.append(torch.tensor([]))
            # else:
            #     t = t.type(torch.long)
            #     t[:, 4] = t[:, 6]
            #     outputs.append(t[:, :5])
        return []


class Predictor(TorchCommander):
    def create_predictor(self):
        model_name = self.weights['model_name']

        if model_name == 'effdet':
            depth = self.weights['args'].depth
            cfg = get_efficientdet_config(f'tf_efficientdet_{depth}')
            cfg.num_classes = 6
            model = EfficientDet(cfg)
            model.load_state_dict(self.weights['state_dict'])
            bench = DetBenchPredict(model).to(self.device)
            bench.eval()
            return EffdetPredictor(bench, model, SIZE_BY_DEPTH[depth])
        elif model_name == 'yolo':
            model = YOLOv3()
            model = model.to(self.device)
            ### WORKAROUND BEGIN
            # dp = os.path.join(os.path.dirname(self.args.weights), str(self.weights['epoch']) + '.darknet')
            # model.load_darknet_weights(dp)
            model.load_state_dict(self.weights['state_dict'])
            ### WORKAROUND END
            return YOLOPredictor(model, self.args.conf_thres, self.args.nms_thres)
        elif model_name == 'yolor':
            model = Yolor().to(self.device)
            # run once
            _ = model(torch.ones([1, 3, 512, 512]).type(torch.FloatTensor).to(self.device))
            model.load_state_dict(self.weights['state_dict'])
            return YolorPredictor(model, self.args.conf_thres, self.args.nms_thres)
        else:
            raise ValueError(f'Invalid model_name: {model_name}')

    def arg_common(self, parser):
        parser.add_argument('-w', '--weights', type=str, required=True)
        parser.add_argument('-b', '--batch-size', type=int, default=16)
        parser.add_argument('-o', '--open', action='store_true')
        parser.add_argument('--conf-thres', type=float, default=0.2)
        parser.add_argument('--nms-thres', type=float, default=0.1, help='iou thresshold for non-maximum suppression')

    def pre_common(self):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)
        self.weights = torch.load(self.args.weights, map_location=lambda storage, loc: storage)

    def detect_images(self, predictor, imgs):
        sizes = [(i.width, i.height) for i in imgs]
        image_size = predictor.get_image_size()
        imgs = [i.resize((image_size, image_size)) for i in imgs]
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4831]*3, [0.3257]*3),
        ])

        bs = self.args.batch_size
        outputs = []
        start = 0
        t = tqdm(range(0, len(imgs), bs))
        for start in t:
            batch = imgs[start:start + bs]

            tt = torch.stack([transform_image(i) for i in batch]).to(self.device)
            with torch.no_grad():
                o = predictor(tt)
            outputs += o
            t.set_description(f'{start} ~ {start + bs} / {len(imgs)}')
            t.refresh()

        results = []
        for img, bboxes in zip(imgs, outputs):
            best_bboxes = []
            for i in LABEL_TO_STR.keys():
                m = bboxes[bboxes[:, 4] == i]
                if len(m) > 0:
                    best_bboxes.append(m[0])
                else:
                    print('missing {LABEL_TO_STR[i]}')

            draw = ImageDraw.Draw(img)
            for i, result in enumerate(best_bboxes):
                bbox = result[:4]
                label = result[4].item()
                draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=self.font, fill='yellow')
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)
            results.append(img)
        return [i.resize(size) for i, size in zip(results, sizes)]


    def arg_dir(self, parser):
        parser.add_argument('-s', '--src-dir', type=str, required=True)
        parser.add_argument('-d', '--dest-dir', type=str, default='tmp')

    def run_dir(self):
        os.makedirs(self.args.dest_dir, exist_ok=True)
        predictor = self.create_predictor()

        imgs = []
        names = []
        for ext in ('png', 'jpg'):
            for p in glob(os.path.join(self.args.src_dir, f'*.{ext}')):
                imgs.append(Image.open(p))
                names.append(os.path.basename(p))

        if len(imgs) < 1:
            print('empty')
            return

        results = self.detect_images(predictor, imgs)

        for name, result in tqdm(zip(names, results)):
            p = os.path.join(self.args.dest_dir, name)
            result.save(p)
        print('done')

    def arg_single(self, parser):
        parser.add_argument('-s', '--src', type=str, required=True)
        parser.add_argument('-d', '--dest-dir', type=str, default='tmp')

    def run_single(self):
        predictor = self.create_predictor()

        dest_dir = self.args.dest_dir
        os.makedirs(dest_dir, exist_ok=True)

        img_size = 512
        img = Image.open(self.args.src)
        img = self.detect_images(predictor, [img])[0]

        # model_id = state['args'].network + '_e' + str(state['epoch'])
        # dest_dir = os.path.join(self.args.dest, model_id)
        file_name = os.path.basename(os.path.abspath(self.args.src))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_name)
        img.save(dest_path)
        print(f'saved {dest_path}')
        if self.args.open:
            os.system(f'xdg-open {dest_path}')


Predictor().run()
