import subprocess
from itertools import combinations
from collections import OrderedDict

import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

from endaaman import Commander


class Experiment(Commander):
    def arg_common(self, parser):
        pass

    def pre_common(self):
        pass

    def arg_models(self, parser):
        parser.add_argument('-e', '--seed', type=int, default=42)

    def run_models(self):
        models = ['gbm', 'nn', 'svm']
        for c in range(1, 4):
            for ii in list(combinations(range(0, 3), c)):
                mm = [models[i] for i in ii]
                params = ' '.join([str(i) for i in mm])
                suffix = '_'.join([str(i) for i in mm])
                # python table.py train -m gbm --no-show-roc
                command = f'python table.py train --no-show-fig -m {params} -s exp_{suffix} --seed {self.args.seed}'
                print(f'RUN: {command}')
                cp = subprocess.run(command, shell=True)
                print(cp)

        print('done.')

    def run_compare_models(self):
        code_names = ['gbm', 'nn', 'svm']
        names = ['LightGBM', 'NN', 'SVM']
        codes = OrderedDict()
        for c in range(1, 4):
            for ii in list(combinations(range(0, 3), c)):
                code = '_'.join([code_names[i] for i in ii])
                name = ' + '.join([names[i] for i in ii])
                codes[code] = name

        df = pd.DataFrame()

        for (code, name) in codes.items():
            o = torch.load(f'out/output_exp_{code}.pt')
            y = o['data']['test']['y']
            pred = o['data']['test']['pred']
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name}: {auc:.3f}')
            row = {'name': name, 'auc': auc}
            for t in ['youden', 'top_left', 'bottom_right']:
                threshold = o['threshold'][t]
                pred = o['data']['test']['pred'] > threshold
                gt = o['data']['test']['y'].values
                cm = metrics.confusion_matrix(gt, pred)
                row[f'sens({t})'] = cm[1, 1] / cm[1].sum()
                row[f'spec({t})'] = cm[0, 0] / cm[0].sum()
            df = df.append(row, ignore_index=True)
        df.to_excel('out/compare_models.xlsx', header=df.columns, index=False, float_format='%.4f')

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig('out/roc_compare.png')
        plt.show()



ex = Experiment()
ex.run()
