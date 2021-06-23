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
        model = EfficientDet(num_classes=1,
                             network=self.args.network,
                             W_bifpn=EFFDET_PARAMS[self.args.network]['W_bifpn'],
                             D_bifpn=EFFDET_PARAMS[self.args.network]['D_bifpn'],
                             D_class=EFFDET_PARAMS[self.args.network]['D_class'])

        img = torch.randn(1, 3, 512, 512)
        # annotations = torch.tensor([[[0, 0, 0, 0, 1, 1]]], dtype=torch.long)
        bb =torch.tensor([[
            # [1, 0, 0, 1, 0.5, 0.4],
            [0, 10, 10, 100, 100, 1], # batch, left, top, right, bottom, id
        ]])
        criterion = FocalLoss()
        classification, regression, anchors = model(img)
        classification_loss, regression_loss = criterion(classification, regression, anchors, bb, 'cpu')
        print(classification_loss)
        print(regression_loss)

        # classification_loss = classification_loss.mean()
        # regression_loss = regression_loss.mean()
        # loss = classification_loss + regression_loss


MyTrainer().run()
