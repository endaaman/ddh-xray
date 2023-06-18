import os
from glob import glob

import torch
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics as skmetrics
import numpy as np
from endaaman.ml import BaseMLCLI, roc_auc_ci


J = os.path.join


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    print(nsamples)
    auc_differences = []
    auc1 = skmetrics.roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = skmetrics.roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = skmetrics.roc_auc_score(y_test.ravel(), p1)
        auc2 = skmetrics.roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)


class CLI(BaseMLCLI):
    class RocArgs(BaseMLCLI.CommonArgs):
        target: str = 'val'

    def run_roc(self, a:RocArgs):
        tt = (
            [
                'Xp + Features',
                'out/classification/full_8/tf_efficientnet_b8_0/',
            ], [
                'Xp',
                'out/classification/full_0/tf_efficientnet_b8_0/',
            ]
        )

        for t in tt:
            name, exp_dir = t
            data = torch.load(J(exp_dir, 'predictions.pt'))
            preds = data['val_preds'].flatten().numpy()
            gts = data['val_gts'].flatten().numpy()
            t.append(preds)
            t.append(gts)
            fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
            auc = skmetrics.auc(fpr, tpr)
            lower, upper = roc_auc_ci(gts, preds)
            plt.plot(fpr, tpr, label=f'{name} AUC:{auc:.3f}({lower:.3f}-{upper:.3f})')

        print(permutation_test_between_clfs(tt[0][3], tt[0][2], tt[1][2]))

        plt.legend()
        plt.savefig('out/roc.png')
        plt.show()

    class CenterCropArgs(BaseMLCLI.CommonArgs):
        size: int

    def run_center_crop(self, a:CenterCropArgs):
        d = f'tmp/crop_{a.size}'
        os.makedirs(d, exist_ok=True)
        pp = glob('data/images/*.jpg')
        for p in tqdm(pp):
            name = os.path.split(p)[-1]
            i = Image.open(p)
            w, h = i.size
            x0 = (w - a.size)/2
            y0 = (h - a.size)/2
            x1 = (w + a.size)/2
            y1 = (h + a.size)/2
            i = i.crop((x0, y0, x1, y1))
            i.save(f'{d}/{name}')



if __name__ == '__main__':
    cli = CLI()
    cli.run()

