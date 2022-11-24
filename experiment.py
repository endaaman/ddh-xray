import subprocess
from itertools import combinations
from collections import OrderedDict

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn import metrics
from adjusttext import adjust_text

from endaaman import Commander


plt.rcParams['font.family'] = 'arial'

class Experiment(Commander):
    def savefig(self, name):
        if self.args.tiff:
            file_name = f'{name}.tiff'
            dpi = 800
        else:
            file_name = f'{name}.png'
            dpi = None

        plt.savefig(file_name, dpi=dpi)
        print(f'saved {file_name}')


    def arg_common(self, parser):
        parser.add_argument('--tiff', action='store_true')

    def pre_common(self):
        pass

    def arg_models(self, parser):
        parser.add_argument('-e', '--seed', type=int, default=34)

    def run_comb(self):
        models = ['gbm', 'nn', 'svm']
        for c in range(1, 4):
            for ii in list(combinations(range(0, 3), c)):
                mm = [models[i] for i in ii]
                params = ' '.join([str(i) for i in mm])
                suffix = '_'.join([str(i) for i in mm])
                # python table.py train -m gbm --no-show-roc
                command = f'python table.py train --no-show-fig -m {params} --suffix exp_{suffix} --seed {self.args.seed}'
                print(f'RUN: {command}')
                cp = subprocess.run(command, shell=True)
                print(cp)

        print('done.')

    def run_compare_comb(self):
        code_names = ['gbm', 'nn', 'svm']
        names = ['LightGBM', 'NN', 'SVM']
        codes = OrderedDict()
        for c in range(1, 4):
            for ii in list(combinations(range(0, 3), c)):
                code = '_'.join([code_names[i] for i in ii])
                name = '+'.join([names[i] for i in ii])
                codes[code] = name


        # plt.rcParams['figure.figsize'] = (12, 12)
        plt.figure(figsize=(7, 5))
        # plt.subplots(figsize=(4, 4))

        df = pd.DataFrame()
        for (i, (code, name)) in enumerate(reversed(codes.items())):
            o = torch.load(f'out/output_exp_{code}.pt')
            y = o['data']['test']['y']
            pred = o['data']['test']['pred']
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name}: AUC={auc:.3f}', zorder=100-i)
            row = {'name': name, 'AUC': auc}
            for t in ['f1', 'youden', 'top-left']:
                threshold = o['threshold'][t]
                pred = o['data']['ind']['pred'] > threshold
                gt = o['data']['ind']['y'].values
                cm = metrics.confusion_matrix(gt, pred)
                row[f'acc({t})'] = (cm[0, 0] + cm[1, 1]) / cm.sum()
                row[f'sens({t})'] = cm[1, 1] / cm[1].sum()
                row[f'spec({t})'] = cm[0, 0] / cm[0].sum()

            df = df.append(row, ignore_index=True)
        df.to_excel('out/compare_models.xlsx', header=df.columns, index=False, float_format='%.4f')

        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('1 - Specificity', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)

        # plt.grid(True)
        self.savefig('out/roc_compare')

        plt.show()

    def arg_thres(self, parser):
        parser.add_argument('--with-arrow', action='store_true')
        parser.add_argument('--abbr', action='store_true')

    def run_thres(self):
        plt.figure(figsize=(7, 5))
        # fig, ax = plt.subplots()

        code = 'gbm_nn_svm'
        name = 'LightGBM+NN+SVM'
        o = torch.load(f'out/output_exp_{code}.pt')
        y = o['data']['train']['y']
        pred = o['data']['train']['pred']
        m = metrics.roc_curve(y, pred)
        fpr, tpr, thresholds = m
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name}: AUC={auc:.3f}')

        f1_scores = [metrics.f1_score(y, pred > t) for t in thresholds]
        idx_f1 = np.argmax(f1_scores)
        threshold_f1 = thresholds[idx_f1]

        idx_youden = np.argmax(tpr - fpr)
        threshold_youden = thresholds[idx_youden]
        idx_topleft = np.argmin((- tpr + 1) ** 2 + fpr ** 2)
        threshold_topleft = thresholds[idx_topleft]

        texts = []

        # youden lines
        x = np.arange(-2, 2, 0.1)
        y = x
        # 45deg line
        plt.plot(x, y, color='black', linewidth=0.5)

        # vertical line
        x_youden = fpr[idx_youden] # fpr
        y_youden = tpr[idx_youden] # tpr
        np.array([x_youden])
        plt.vlines(x_youden, x_youden, y_youden, color='black', linestyle='dashed', linewidth=1)

        youden_index = y_youden - x_youden
        texts += [
            plt.text(x_youden, y_youden-0.2, f'Youden index={youden_index:.3f}', fontstyle='italic', color='black')
        ]

        print(f'f1 {threshold_f1:.3f} youden {threshold_youden:.3f} topleft {threshold_topleft:.3f}')

        x_topleft = fpr[idx_topleft]
        y_topleft = tpr[idx_topleft]
        if self.args.with_arrow:
            # topleft arrow
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
            [y_topleft, y_youden, y_f1], marker='x', color='red', zorder=100)

        if self.args.abbr:
            labels = ['T1', 'T2', 'T3']
        else:
            labels = [
                f'{s} threshold={th:.3f}'
                for (th, s) in (
                    (threshold_f1, 'f1'),
                    (threshold_youden, 'Youden'),
                    (threshold_topleft,  'top-left'),
                )
            ]

        poss = (
            (x_f1, y_f1),
            (x_youden, y_youden),
            (x_topleft, y_topleft),
        )
        offsets = (-0.02, 0.02) if self.args.abbr else (0, 0)

        texts += [
            plt.text(x+offsets[0], y+offsets[1], label, fontstyle='italic', color='black')
            for i, ((x, y), label) in enumerate(zip(poss, labels))
        ]
        for t in texts:
            t.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        if not self.args.abbr:
            adjust_text(texts)

        offset = 0.05
        plt.xlim([-offset, 1+offset])
        plt.ylim([-offset, 1+offset])

        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('1 - Specificity', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        # plt.grid(True)
        self.savefig(f'out/roc_thres{self.get_suffix()}')
        plt.show()

    def run_roc_acetabular(self):
        df = pd.read_excel('data/ROC foracetabular dysplasia.xlsx')
        print(df)

        gt = - df['gt'].values + 1
        pred = df['probabiility']
        fpr, tpr, thresholds = metrics.roc_curve(gt, pred)

        auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')

        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('1 - Specificity', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        # plt.grid(True)
        self.savefig('out/roc_acetabular')
        plt.show()

ex = Experiment(defaults={'seed': 34})
ex.run()
