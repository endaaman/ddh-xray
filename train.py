import json
from PIL import Image
import torch
from torchvision import transforms

from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from endaaman import Trainer


class MyTrainer(Trainer):

    def arg_common(self, parser):
        parser.add_argument('--network', default='efficientdet-d0', type=str, help='efficientdet-[d0, d1, ..]')

    def run_train(self):
        model = EfficientDet(
            num_classes=1,
            network=self.args.network,
            W_bifpn=EFFDET_PARAMS[self.args.network]['W_bifpn'],
            D_bifpn=EFFDET_PARAMS[self.args.network]['D_bifpn'],
            D_class=EFFDET_PARAMS[self.args.network]['D_class'])

        img = torch.randn(2, 3, 512, 512)
        # img = Image.open('data/misc/piece.jpg')
        # img = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])(img)
        # img = img[None, :, : ,:]

        # result = model(img)
        # print('class:', result[1][0])

        bb = torch.tensor([
            [
                # [1, 0, 0, 1, 0.5, 0.4],
                [0, 1, 2, 3, 4], # id, left, top, right, bottom
                [1, 1, 2, 3, 4], # id, left, top, right, bottom
            ],
            [
                [0, 1, 2, 3, 4], # id, left, top, right, bottom
                [0, 1, 2, 3, 4], # id, left, top, right, bottom
           ],
        ])

        criterion = FocalLoss()
        classification, regression, anchors = model(img)
        print('anchors shape:', anchors.shape)
        print('classification shape:', classification.shape)
        print('regression shape:', regression.shape)
        classification_loss, regression_loss = criterion(classification, regression, anchors, bb, 'cpu')
        print(classification_loss)
        print(regression_loss)


MyTrainer().run()
