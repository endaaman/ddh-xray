import os
import json

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import pil_to_tensor
from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from augmentation import ResizeAugmentation
from datasets import XrayDataset
from endaaman import Trainer


class MyEval(Trainer):
    def create_model(self, network):
        network = f'efficientdet-{network}'
        model = EfficientDet(
            num_classes=6,
            network=network,
            W_bifpn=EFFDET_PARAMS[network]['W_bifpn'],
            D_bifpn=EFFDET_PARAMS[network]['D_bifpn'],
            D_class=EFFDET_PARAMS[network]['D_class'])
        return model.to(self.device)

    def arg_single(self, parser):
        parser.add_argument('-w', '--weights', type=str, required=True)
        parser.add_argument('-s', '--src', type=str, required=True)
        parser.add_argument('-d', '--dest', type=str, default='out')
        parser.add_argument('-o', '--open', action='store_true')

    def run_single(self):
        state = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        model = self.create_model(state['args'].network)
        model.load_state_dict(state['state_dict'])

        img_size = 640

        img = Image.open(self.args.src)
        img = img.resize((img_size, img_size))
        img_tensor = pil_to_tensor(img)[None, :, :, :]


        model.eval()
        with torch.no_grad():
            scores, classification, transformed_anchors = model.infer(img_tensor)
            print(scores.shape, classification.shape, transformed_anchors.shape)

        bboxes = []
        # labels = []
        bbox_scores = []
        draw = ImageDraw.Draw(img)
        for j in range(scores.shape[0]):
            bbox = transformed_anchors[[j], :][0].data.cpu().numpy()
            scale = img_size / 512
            bbox = (bbox * scale).astype(np.int)
            bboxes.append(list(bbox))
            # label_name = VOC_CLASSES[int(classification[[j]])]
            # labels.append(label_name)
            # cv2.rectangle(origin_img, (x1, y1), (x2, y2), (179, 255, 179), 2, 1)

            clas = int(classification[[j]])
            score = np.around(scores[[j]].cpu().numpy(), decimals=2) * 100
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='yellow', width=1)
            if score > 30:
                print(score, clas, bbox)
            # labelSize, baseLine = cv2.getTextSize('{} {}'.format(
            #     label_name, int(score)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # cv2.rectangle(
            #     origin_img, (x1, y1-labelSize[1]), (x1+labelSize[0], y1+baseLine), (223, 128, 255), cv2.FILLED)
            # cv2.putText(
            #     origin_img, '{} {}'.format(label_name, int(score)),
            #     (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.8, (0, 0, 0), 2
            # )
            # bbox_scores.append(int(score))

        file_name = os.path.basename(os.path.abspath(self.args.src))
        model_id = state['args'].network + '_e' + str(state['epoch'])
        dest_dir = os.path.join(self.args.dest, model_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_name)
        img.save(dest_path)
        print(f'saved {dest_path}')
        if self.args.open:
            os.system(f'xdg-open {dest_path}')


MyEval().run()
