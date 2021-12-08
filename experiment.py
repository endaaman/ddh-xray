import subprocess
from itertools import combinations
from collections import OrderedDict

import torch
from matplotlib import pyplot as plt
from sklearn import metrics

from endaaman import Commander


class Experiment(Commander):
    def arg_common(self, parser):
        pass

    def pre_common(self):
        pass

    def arg_comb(self, parser):
        parser.add_argument('-e', '--seed', type=int, default=42)

    def run_comb(self):
        models = ['gbm', 'nn', 'svm']
        for c in range(1, 4):
            for ii in list(combinations(range(0, 3), c)):
                mm = [models[i] for i in ii]
                params = ' '.join([str(i) for i in mm])
                suffix = '_'.join([str(i) for i in mm])
                # python table.py train -m gbm --no-show-roc
                command = f'python table.py train --no-show-roc -m {params} -s exp_{suffix} --seed {self.args.seed}'
                cp = subprocess.run(command, shell=True)
                print(cp)

        print('done.')

    def run_compare_roc(self):
        models = ['gbm', 'nn', 'svm']
        ss = []
        for c in range(1, 4):
            for ii in list(combinations(range(0, 3), c)):
                ss.append('_'.join([models[i] for i in ii]))

        outputs = OrderedDict()
        for s in ss:
            outputs[s] = torch.load(f'out/output_exp_{s}.pt')
        self.outputs = outputs

        for (s, o) in outputs.items():
            y = o['data']['test']['y']
            pred = o['data']['test']['pred']
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{s}: {auc:.3f}')

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.show()



ex = Experiment()
ex.run()
