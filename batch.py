import os

from matplotlib import pyplot as plt
from sklearn import metrics as skmetrics
import numpy as np
from endaaman.ml import BaseMLCLI, roc_auc_ci


J = os.path.join


class CLI(BaseMLCLI):
    class RocArgs(BaseMLCLI.CommonArgs):
        target: str = 'val'

    def run_roc(self, a:RocArgs):
        targets = (
            (
                'Xp + Features',
                'out/classification/full_8/tf_efficientnet_b0',
            ),
            (
               'Features',
               'out/classification/feature_8/linear',
            ),
            (
               'Xp',
               'out/classification/full_0/tf_efficientnet_b0',
            ),
        )

        for (name, exp_dir) in targets:
            print(name, exp_dir)
            # preds = np.load(J(exp_dir, 'val_preds.npy'))
            # gts = np.load(J(exp_dir, 'val_gts.npy'))
            preds = np.load(J(exp_dir, f'{a.target}_preds.npy'))
            gts = np.load(J(exp_dir, f'{a.target}_gts.npy'))
            fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
            auc = skmetrics.auc(fpr, tpr)
            lower, upper = roc_auc_ci(gts, preds)
            plt.plot(fpr, tpr, label=f'{name} AUC:{auc:.3f}({lower:.3f}-{upper:.3f})')

        plt.legend()
        plt.savefig('out/roc.png')



if __name__ == '__main__':
    cli = CLI()
    cli.run()

