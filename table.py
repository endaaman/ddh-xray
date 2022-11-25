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

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import optuna

from endaaman.torch import MLCommander

from bench import LightGBMBench, XGBBench, SVMBench, NNBench
from datasets.table import cols_cat, col_target, cols_feature, cols_extend, col_to_label


optuna.logging.disable_default_handler()


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

class Table(MLCommander):
    def arg_common(self, parser):
        parser.add_argument('-r', '--test-ratio', type=float)
        parser.add_argument('-o', '--optuna', action='store_true')
        parser.add_argument('--dropna', action='store_true')
        parser.add_argument('--a-flip', action='store_true')
        parser.add_argument('--a-fill', action='store_true')

    def pre_common(self):
        df_table = pd.read_excel('data/table.xlsx', index_col=0) #len:765
        df_measure = pd.read_excel('data/measurement_all.xlsx', index_col=0) #len:763

        df_all = df_table.merge(df_measure,left_index=True, right_index=True, how='outer')

        if self.args.dropna:
            df_all = df_all.dropna()

        # for col, fn in cols_extend.items():
        #     print(col, fn)
        #     df_all[col] = fn(df_all)

        if self.args.test_ratio:
            df_train, df_test = train_test_split(
                df_all,
                test_size=self.args.test_ratio,
                random_state=self.args.seed,
                stratify=df_all['treatment'])

            df_train['test'] = 0
            df_test['test'] = 1
        else:
            df_train = df_all[df_all['test'] == 0]
            df_test = df_all[df_all['test'] == 1]


        df_table_ind = pd.read_excel('data/table_independent.xlsx', index_col=0)
        df_measure_ind = pd.read_excel('data/measurement_independent.xlsx', index_col=0)
        df_ind = pd.concat([df_table_ind, df_measure_ind], axis=1)

        df_manual = pd.read_excel('data/manual_ind.xlsx', index_col=0)

        if self.args.a_fill:
            fill_by_opposite(df_train)
            fill_by_opposite(df_test)
            fill_by_opposite(df_ind)

        if self.args.a_flip:
            df_train_mirror = df_train.copy()
            df_train_mirror[['left_a', 'right_a']] = df_train[['right_a', 'left_a']]
            df_train_mirror[['left_b', 'right_b']] = df_train[['right_b', 'left_b']]
            df_train_mirror[['left_oe', 'right_oe']] = df_train[['right_oe', 'left_oe']]
            df_train_mirror[['left_alpha', 'right_alpha']] = df_train[['right_alpha', 'left_alpha']]
            df_train = pd.concat([df_train, df_train_mirror])

        t = cols_feature + [col_target]
        self.df_all = df_all[t]
        self.df_train = df_train[t]
        self.df_test = df_test[t]
        self.df_manual = df_manual[t]
        self.df_ind = df_ind[t]

        # self.meta_model = LogisticRegression()
        self.meta_model = LinearRegression()

    def run_demo(self):
        print(len(self.df_test))
        print(len(self.df_train))

    def create_benchs_by_args(self, args):
        t = {
            'gbm': lambda: LightGBMBench(
                num_folds=args.folds,
                seed=args.seed,
                imputer=args.imputer,
                use_optuna=args.optuna),
            'xgb': lambda: XGBBench(
                num_folds=args.folds,
                seed=args.seed,
            ),
            'svm': lambda: SVMBench(
                num_folds=args.folds,
                seed=args.seed,
                imputer=args.imputer,
                svm_kernel=args.kernel,
            ),
            'nn': lambda: NNBench(
                num_folds=args.folds,
                seed=args.seed,
                epoch=200,
            ),
        }
        return [t[m]() for m in args.models]

    def arg_train(self, parser):
        parser.add_argument('--no-show-fig', action='store_true')
        parser.add_argument('--folds', type=int, default=5)
        parser.add_argument('-i', '--imputer')
        parser.add_argument('-m', '--models', default=['gbm'], nargs='+', choices=['gbm', 'xgb', 'svm', 'nn'])
        parser.add_argument('-k', '--kernel', default='rbf')
        parser.add_argument('-g', '--gather', default='median', choices=['median', 'mean', 'reg'])
        parser.add_argument('-b', '--mean-by-bench', action='store_true')

    def predict_benchs(self, benchs, x):
        if self.args.mean_by_bench:
            return np.stack([np.mean(b.predict(x), axis=1) for b in benchs], axis=1)
        return np.concatenate([b.predict(x) for b in benchs], axis=1)

    def run_train(self):
        benchs = self.create_benchs_by_args(self.args)

        for b in benchs:
            b.train(self.df_train, col_target)

        # p = f'out/model{self.self.args.suffix}.txt'
        # model.save_model(p)
        checkpoint = {
            'args': self.args,
            'model_data': [b.serialize() for b in benchs],
        }
        p = f'out/model{self.args.suffix}.pth'
        torch.save(checkpoint, p)
        print(f'wrote {p}')

        x_train = self.df_train[cols_feature]
        y_train = self.df_train[col_target]
        preds_train = self.predict_benchs(benchs, x_train)
        self.meta_model.fit(preds_train, y_train)

        for b in benchs:
            print(type(b).__name__, 'train %.2f pred %.2f' % (b.training_time, b.predicting_time))
        print()

        self.evaluate(benchs, not self.args.no_show_fig)

    def evaluate(self, benchs, show_fig):
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
            'manual': {
                'x': self.df_manual[cols_feature],
                'y': self.df_manual[col_target],
            },
        }

        for (t, v) in data.items():
            preds = self.predict_benchs(benchs, v['x'])
            if self.args.gather == 'median':
                pred = np.median(preds, axis=1)
            elif self.args.gather == 'mean':
                pred = np.mean(preds, axis=1)
            elif self.args.gather == 'reg':
                pred = self.meta_model.predict(preds)
            else:
                raise RuntimeError(f'Invalid gather rule: {self.args.gather}')
            v['pred'] = pred

        threshold = OrderedDict()
        result = {}
        for t in ['train', 'test', 'ind']:
            y = data[t]['y']
            pred = data[t]['pred']
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            if t == 'train':
                # f1
                f1_scores = [metrics.f1_score(y, pred > t) for t in thresholds]
                threshold['f1'] = thresholds[np.argmax(f1_scores)]

                # youden
                sums = tpr - fpr
                threshold['youden'] = thresholds[np.argmax(sums)]

                # top-left
                sums = (- tpr + 1) ** 2 + fpr ** 2
                threshold['top-left'] = thresholds[np.argmin(sums)]

                # bottom-right
                # sums = tpr ** 2 + (-fpr + 1) ** 2
                # threshold['bottom_right'] = thresholds[np.argmax(sums)]
            else:
                pass
                # print(f'test tpr: {tpr[best_index]}')
                # print(f'test fpr: {fpr[best_index]}')
            auc = metrics.auc(fpr, tpr)
            result[t] = {
                'thresholds': thresholds,
                'tpr': tpr,
                'fpr': fpr,
            }
            plt.plot(fpr, tpr, label=f'{t} auc={auc:.3f}')

        output ={
            'threshold': threshold,
            'result': result,
            'data': data,
        }

        self.output = output

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        if show_fig:
            plt.show()

        plt.savefig(f'out/roc{self.args.suffix}.png')
        plt.close()

        # targets = ['test', 'ind', 'manual']
        targets = ['ind']

        fig = plt.figure(figsize=(16, len(targets)*5), constrained_layout=True)
        # fig.suptitle(f'Models: {" + ".join(self.args.models)}')

        for col, (n, th) in enumerate(threshold.items()):
            print(f'threshold {n}: {th:.4f}')
            for row, t in enumerate(targets):
                pred = data[t]['pred'] > th
                gt = data[t]['y'].values > 0.1
                cm = metrics.confusion_matrix(gt, pred)
                print(f'{t}: ', cm)

                ax = fig.add_subplot(len(targets), len(threshold.keys()), row*3+col+1)
                ax.matshow(cm, cmap=plt.cm.GnBu)
                for i in range(cm.shape[1]):
                    for j in range(cm.shape[0]):
                        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size=14)
                sens = cm[1, 1] / cm[1].sum()
                spec = cm[0, 0] / cm[0].sum()

                ax.set_ylabel('Ground truth', fontsize=14)
                ax.set_xlabel('Prediction', fontsize=14)
                ax.xaxis.set_label_position('bottom')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_title(f'{n} threshold={th:.3f}\nsensitivity:{sens:.3f} specificity:{spec:.3f}', fontsize=14)

        # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
        # plt.tight_layout()
        plt.savefig(f'out/cm{self.args.suffix}.png')
        if show_fig:
            plt.show()

        output_path = f'out/output{self.args.suffix}.pt'
        torch.save(output, output_path)
        print(f'wrote output to {output_path}')

    def arg_roc(self, parser):
        parser.add_argument('-c', '--checkpoint')

    def run_roc(self):
        checkpoint = torch.load(self.args.checkpoint)
        train_args = checkpoint['args']
        bench = self.create_bench_by_args(self.args)
        bench.restore(checkpoint['model_data'])
        self.evaluate(bench, True)

    def arg_violin(self, parser):
        parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])

    def run_violin(self):
        df = self.df_train if self.args.mode == 'train' else self.df_test
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        fig.suptitle(f'Data distribution ({self.args.mode} dataset)')
        for i, col in enumerate(['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b']):
            ax = axes[i//4][i%4]
            ax.set_title(col)
            # sns.violinplot(x='female', y=col, hue='treatment', data=self.df_train, split=True, ax=ax)
            sns.violinplot(y=col, x='treatment', hue='female', data=df, split=True, ax=ax)

        p = f'tmp/violin_{self.args.mode}.png'
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


runner = Table()
runner.run()
