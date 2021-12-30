import subprocess
from itertools import combinations
from collections import OrderedDict

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from adjustText import adjust_text

from endaaman import Commander


class Experiment(Commander):
    def arg_common(self, parser):
        pass

    def pre_common(self):
        pass

    def arg_models(self, parser):
        parser.add_argument('-e', '--seed', type=int, default=34)

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

        plt.rcParams['font.family'] = 'arial'
        # plt.rcParams['figure.figsize'] = (12, 12)
        plt.figure(figsize=(8, 6))
        # plt.subplots(figsize=(4, 4))

        for (i, (code, name)) in enumerate(reversed(codes.items())):
            o = torch.load(f'out/output_exp_{code}.pt')
            y = o['data']['test']['y']
            pred = o['data']['test']['pred']
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name}: AUC={auc:.3f}', zorder=100-i)
            row = {'name': name, 'auc': auc}
            for t in ['f1', 'youden', 'top_left', 'bottom_right']:
                threshold = o['threshold'][t]
                pred = o['data']['test']['pred'] > threshold
                gt = o['data']['test']['y'].values
                cm = metrics.confusion_matrix(gt, pred)
                row[f'sens({t})'] = cm[1, 1] / cm[1].sum()
                row[f'spec({t})'] = cm[0, 0] / cm[0].sum()
            df = df.append(row, ignore_index=True)
        df.to_excel('out/compare_models.xlsx', header=df.columns, index=False, float_format='%.4f')

        plt.legend(loc='lower right')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig('out/roc_compare.png')
        plt.show()

    def run_thres(self):
        plt.figure(figsize=(8, 6))
        # fig, ax = plt.subplots()

        code = 'gbm_nn_svm'
        name = 'LightGBM + NN + SVM'
        o = torch.load(f'out/output_exp_{code}.pt')
        y = o['data']['test']['y']
        pred = o['data']['test']['pred']
        m = metrics.roc_curve(y, pred)
        fpr, tpr, thresholds = m
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name}: AUC={auc:.3f}')

        f1_scores = [metrics.f1_score(y, pred > t) for t in thresholds]
        idx_f1 = np.argmax(f1_scores)
        threshold_f1 = thresholds[idx_f1]

        idx_youden = np.argmax(tpr + 1 - fpr)
        threshold_youden = thresholds[idx_youden]
        idx_topleft = np.argmin((- tpr + 1) ** 2 + fpr ** 2)
        threshold_topleft = thresholds[idx_topleft]

        # youden lines
        x = np.arange(-2, 2, 0.1)
        y = x
        # 45deg line
        plt.plot(x, y, color='black', linewidth=0.5)

        # vertical line
        x_youden = fpr[idx_youden]
        y_youden = tpr[idx_youden]
        np.array([x_youden])
        plt.vlines(x_youden, x_youden, y_youden, color='black', linestyle='dashed', linewidth=1)


        print(f'f1 {threshold_f1:.3f} youen {threshold_youden:.3f} topleft {threshold_topleft:.3f}')

        # topleft arrow
        x_topleft = fpr[idx_topleft]
        y_topleft = tpr[idx_topleft]
        plt.annotate('',
                    xy=(x_topleft, y_topleft),
                    xytext=(0, 1),
                    arrowprops=dict(
                        shrink=0,
                        width=0.5,
                        headwidth=3,
                        headlength=4,
                        facecolor='gray',
                        edgecolor='gray')
                    )


        # f1
        x_f1 = fpr[idx_f1]
        y_f1 = tpr[idx_f1]

        # point
        plt.scatter(
            [x_topleft, x_youden, x_f1],
            [y_topleft, y_youden, y_f1], marker='x', color='darkorange', zorder=100)

        texts = [
            plt.text(x_youden, y_youden, f'youden threshold={threshold_youden:.3f}', fontstyle='italic'),
            plt.text(x_topleft, y_topleft, f'topleft threshold={threshold_topleft:.3f}', fontstyle='italic'),
            plt.text(x_f1, y_f1, f'f1 threshold={threshold_f1:.3f}', fontstyle='italic'),
        ]
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        adjust_text(texts)

        offset = 0.05
        plt.xlim([-offset, 1+offset])
        plt.ylim([-offset, 1+offset])

        plt.legend(loc='lower right')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig('out/roc_thres.png')
        plt.show()


ex = Experiment()
ex.run()
