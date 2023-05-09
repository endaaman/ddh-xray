import random
import os
import copy
from collections import OrderedDict

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import scipy.stats as st
from pydantic import Field

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

from endaaman.ml import BaseMLCLI

from bench import LightGBMBench, SVMBench, NNBench
from common import load_data, col_target, cols_feature, cols_extend, col_to_label



def fill_by_opposite(df):
    for i in df.index:
        for v in ['a', 'b', 'oe', 'alpha']:
            left = np.isnan(df[f'left_{v}'][i])
            right = np.isnan(df[f'right_{v}'][i])
            if not left ^ right:
                continue
            if left:
                df[f'left_{v}'][i] = df[f'right_{v}'][i]
                # print(f'fill: [{i}] right {v}')
            if right:
                df[f'left_{v}'][i] = df[f'right_{v}'][i]
                # print(f'fill: [{i}] left {v}')

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        test_ratio:float = Field(-1, cli=('-r', '--test-ratio', ))
        normalize:bool = Field(False, cli=('--normalize', ))

    def pre_common(self, a:CommonArgs):
        dfs = load_data(test_ratio=a.test_ratio, normalize_features=a.normalize, seed=a.seed)
        df_all = dfs['all']
        df_train = dfs['train']
        df_test = dfs['test']
        df_ind = dfs['ind']

        t = cols_feature + [col_target]
        self.df_all = df_all[t]
        self.df_train = df_train[t]
        self.df_test = df_test[t]
        self.df_ind = df_ind[t]

        # self.meta_model = LogisticRegression()
        self.meta_model = LinearRegression()

    def run_demo(self, a:CommonArgs):
        print(len(self.df_test))
        print(len(self.df_train))

    def create_benchs(self, models:list[str], num_folds, seed):
        factory = {
            'gbm': lambda: LightGBMBench(
                num_folds=num_folds,
                seed=seed),
            'svm': lambda: SVMBench(
                num_folds=num_folds,
                seed=seed,
                svm_kernel='rbf',
            ),
            'nn': lambda: NNBench(
                num_folds=num_folds,
                seed=seed,
                epoch=200,
            ),
        }
        return [factory[m]() for m in models]

    class TrainArgs(CommonArgs):
        no_show_fig:bool = Field(False, cli=('--no-show-fig', ))
        num_folds:int = Field(5, cli=('--folds', ))
        models_base:str = Field('gbm', cli=('--models', '-m'), choices=['gbm', 'svm', 'nn'])
        gather:str = 'median' # choices=['median', 'mean', 'reg']
        mean_by_benches:bool = Field(False, cli=('--mean-by-benchs', '-M'))
        threshold_type:str = Field('youden', cli=('--threshold', '-t'), choices=['youden', 'f1', 'topleft'])

        @property
        def models(self):
            return sorted(self.models_base.split('_'))

    def predict_benchs(self, benchs, x, mean_by_benches):
        if mean_by_benches:
            return np.stack([np.mean(b.predict(x), axis=1) for b in benchs], axis=1)
        return np.concatenate([b.predict(x) for b in benchs], axis=1)

    def run_train(self, a:TrainArgs):
        code = '_'.join(a.models)
        benchs = self.create_benchs(a.models, a.num_folds, a.seed)
        for b in benchs:
            b.train(self.df_train, col_target)
        x_train = self.df_train[cols_feature]
        y_train = self.df_train[col_target]
        preds_train = self.predict_benchs(benchs, x_train, a.mean_by_benches)
        self.meta_model.fit(preds_train, y_train)

        for b in benchs:
            print(type(b).__name__, 'train %.2f pred %.2f' % (b.training_time, b.predicting_time))

        data = {
            'train': {
                'x': self.df_train[cols_feature],
                'y': self.df_train[col_target],
            },
            'test': {
                'x': self.df_test[cols_feature],
                'y': self.df_test[col_target],
            },
            'ind': {
                'x': self.df_ind[cols_feature],
                'y': self.df_ind[col_target],
            },
            # 'manual': {
            #     'x': self.df_manual[cols_feature],
            #     'y': self.df_manual[col_target],
            # },
        }

        for (t, v) in data.items():
            preds = self.predict_benchs(benchs, v['x'], a.mean_by_benches)
            if a.gather == 'median':
                pred = np.median(preds, axis=1)
            elif a.gather == 'mean':
                pred = np.mean(preds, axis=1)
            elif a.gather == 'reg':
                pred = self.meta_model.predict(preds)
            else:
                raise RuntimeError(f'Invalid gather rule: {a.gather}')
            v['pred'] = pred

        threshold = None
        results = {}
        for t in data:
            y = data[t]['y']
            pred = data[t]['pred']
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            if t == 'train':
                match a.threshold_type:
                    case 'f1':
                        threshold = thresholds[np.argmax([metrics.f1_score(y, pred > t) for t in thresholds])]
                    case 'youden':
                        threshold = thresholds[np.argmax(tpr - fpr)]
                    case 'topleft':
                        threshold = thresholds[np.argmin((- tpr + 1) ** 2 + fpr ** 2)]
                    case _:
                        raise RuntimeError(f'Invalid threshold_type: {a.threshold_type}')
            auc = metrics.auc(fpr, tpr)
            results[t] = {'tpr': tpr, 'fpr': fpr}
            plt.plot(fpr, tpr, label=f'{t} auc={auc:.3f}')
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        if not a.no_show_fig:
            plt.show()
        plt.savefig(f'out/roc_{code}.png')
        plt.close()

        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        for i, (t, d) in enumerate(data.items()):
            ax = fig.add_subplot(1, len(data), i+1)
            gt = d['y']
            pred = d['pred']
            cm = metrics.confusion_matrix(gt, pred > threshold)
            sns.heatmap(cm, cbar=True, annot=True, fmt='g', ax=ax)
            sens = cm[1, 1] / cm[1].sum()
            spec = cm[0, 0] / cm[0].sum()
            ax.set_ylabel('Ground truth', fontsize=14)
            ax.set_xlabel('Prediction', fontsize=14)
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title(f'{t} recall:{sens:.2f} spec:{spec:.2f}', fontsize=14)

        plt.savefig(f'out/cm_{code}.png')
        if not a.no_show_fig:
            plt.show()

    class ViolinArgs(CommonArgs):
        mode: str = 'train'

    def run_violin(self, a:ViolinArgs):
        df = self.df_train if a.mode == 'train' else self.df_test
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        fig.suptitle(f'Data distribution ({a.mode} dataset)')
        for i, col in enumerate(['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b']):
            ax = axes[i//4][i%4]
            ax.set_title(col)
            # sns.violinplot(x='female', y=col, hue='treatment', data=self.df_train, split=True, ax=ax)
            sns.violinplot(y=col, x='female', hue='treatment', data=df, split=True, ax=ax)

        p = f'tmp/violin_{a.mode}.png'
        plt.savefig(p)
        print(f'wrote {p}')
        plt.show()

    def run_demographic(self):
        m = []

        dfs = OrderedDict((
            ('test', self.df_test),
            ('train', self.df_train),
            ('ind', self.df_ind),
        ))

        bool_keys = ['female', 'breech_presentation', 'treatment']
        for col in bool_keys:
            texts = []
            for (name, df) in dfs.items():
                a = 100 * df[col].sum() / len(df[col])
                b = 100 - a
                v = f'{a:.1f}% : {b:.1f}%'
                if name != 'test':
                    t, p = st.ttest_ind(df[col].values, dfs['test'][col].values, equal_var=False)
                    v += f'(p={p:.3f})'
                texts.append(v)
            print(f'{col}: ' + ' '.join(texts))
            m.append([col] + texts)

        float_keys = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b']
        for col in float_keys:
            texts = []
            for (name, df) in dfs.items():
                mean = df[col].mean()
                std = df[col].std(ddof=1) / np.sqrt(np.size(df[col]))
                v = f'{mean:.2f}±{std:.2f}'
                if name != 'test':
                    t, p = st.ttest_ind(df[col].dropna().values, dfs['test'][col].dropna().values, equal_var=False)
                    v += f'(p={p:.3f})'
                texts.append(v)
            print(f'{col:<8}: ' + ' '.join(texts))
            m.append([col] + texts)
        self.m = m
        o = pd.DataFrame(m)

        header = [f'{label} ({len(df)} cases)' for label, (name, df) in zip(['test', 'train', 'independent'], dfs.items())]
        o.to_excel('out/demographic.xlsx', index=False, header=['column'] + header)

    def run_cm(self):
        cm =  np.array([[38,  8], [ 0,  2]])

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 2, 2)
        ax.matshow(cm, cmap=plt.cm.GnBu)
        for i in range(cm.shape[1]):
            for j in range(cm.shape[0]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', )
        sens = cm[1, 1] / cm[1].sum()
        spec = cm[0, 0] / cm[0].sum()
        ax.set_ylabel('Ground truth')
        ax.set_xlabel('Prediction')
        ax.set_title(f'Sens: {sens:.2f} Spec: {spec:.2f}')
        plt.show()

    def arg_corr(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test', 'ind'])

    def run_corr(self):
        df = {
            'all': self.df_all,
            'train': self.df_train,
            'test': self.df_test,
            'ind': self.df_ind,
        }[self.args.target]

        plt.figure(figsize=(12, 9))
        # plt.rcParams['figure.subplot.bottom'] = 0.3
        # plt.rcParams['lines.linewidth'] = 3
        df = df.rename(columns=col_to_label)
        ax = sns.heatmap(df.corr(), vmax=1, vmin=-1, center=0, annot=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
        plt.subplots_adjust(bottom=0.15, left=0.2)
        plt.savefig(f'out/corr_{self.args.target}.png')
        plt.show()


cli = CLI()
cli.run()
