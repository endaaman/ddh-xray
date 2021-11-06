import random
import os
import copy
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import optuna

from endaaman import Commander
from bench import LightGBMBench, XGBBench, SVMBench, NNBench
from datasets import cols_cat, col_target, cols_feature, cols_extend


optuna.logging.disable_default_handler()



class Table(Commander):
    def get_suffix(self):
        return '_' + self.args.suffix if self.args.suffix else ''

    def arg_common(self, parser):
        parser.add_argument('-r', '--test-ratio', type=float)
        parser.add_argument('-e', '--seed', type=int, default=42)
        parser.add_argument('-s', '--suffix')
        parser.add_argument('-o', '--optuna', action='store_true')

    def pre_common(self):
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

        df = pd.read_excel('data/table.xlsx', index_col=0)
        df_train = df[df['test'] == 0]
        df_test = df[df['test'] == 1]

        df_measure_train = pd.read_excel('data/measurement_train.xlsx', index_col=0)
        df_measure_test = pd.read_excel('data/measurement_test.xlsx', index_col=0)
        df_train = pd.concat([df_train, df_measure_train], axis=1)
        df_test = pd.concat([df_test, df_measure_test], axis=1)

        # self.meta_model = LogisticRegression()
        self.meta_model = LinearRegression()

        self.df_all = pd.concat([df_train, df_test])
        # self.df_all.loc[:, cols_cat]= self.df_all[cols_cat].astype('category')

        for col, fn in cols_extend.items():
            self.df_all[col] = fn(self.df_all)

        if self.args.test_ratio:
            self.df_train, self.df_test = train_test_split(self.df_all, test_size=self.args.test_ratio, random_state=self.args.seed)
        else:
            self.df_train = self.df_all[self.df_all['test'] == 0]
            self.df_test = self.df_all[self.df_all['test'] == 1]

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
                imputer=args.imputer,
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
        return [t[m]() for m in args.model]

    def arg_train(self, parser):
        parser.add_argument('--roc', action='store_true')
        parser.add_argument('--folds', type=int, default=5)
        parser.add_argument('-i', '--imputer')
        parser.add_argument('-m', '--model', default=['gbm'], nargs='+', choices=['gbm', 'xgb', 'svm', 'nn'])
        parser.add_argument('-k', '--kernel', default='rbf')
        parser.add_argument('-g', '--gather', default='median', choices=['mean', 'median', 'reg'])
        parser.add_argument('-b', '--mean-by-bench', action='store_true')

    def predict_benchs(self, benchs, x):
        if self.args.mean_by_bench:
            return np.stack([np.mean(b.predict(x), axis=1) for b in benchs], axis=1)
        else:
            return np.concatenate([b.predict(x) for b in benchs], axis=1)

    def run_train(self):
        benchs = self.create_benchs_by_args(self.args)

        for b in benchs:
            b.train(self.df_train)

        # p = f'out/model{self.get_suffix()}.txt'
        # model.save_model(p)
        checkpoint = {
            'args': self.args,
            'model_data': [b.serialize() for b in benchs],
        }
        p = f'out/model{self.get_suffix()}.pth'
        torch.save(checkpoint, p)
        print(f'wrote {p}')


        # train meta model
        x_train = self.df_train[cols_feature]
        y_train = self.df_train[col_target]
        preds_train = self.predict_benchs(benchs, x_train)
        self.meta_model.fit(preds_train, y_train)

        if self.args.roc:
            self.draw_roc(benchs)

    def draw_roc(self, benchs):
        x_train = self.df_train[cols_feature]
        y_train = self.df_train[col_target]
        x_test = self.df_test[cols_feature]
        y_test = self.df_test[col_target]

        preds_train = self.predict_benchs(benchs, x_train)
        preds_test = self.predict_benchs(benchs, x_test)

        if self.args.gather == 'mean':
            pred_train = np.mean(preds_train, axis=1)
            pred_test = np.mean(preds_test, axis=1)
        elif self.args.gather == 'median':
            pred_train = np.median(preds_train, axis=1)
            pred_test = np.median(preds_test, axis=1)
        elif self.args.gather == 'reg':
            pred_train = self.meta_model.predict(preds_train)
            pred_test = self.meta_model.predict(preds_test)
        else:
            raise RuntimeError(f'Invalid gather rule: {self.args.gather}')

        for (t, y, pred) in (('train', y_train, pred_train), ('test', y_test, pred_test)):
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{t} auc = {auc:.2f}')

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        p = f'out/roc{self.get_suffix()}.png'
        plt.savefig(p)
        plt.show()
        print(f'wrote {p}')

    def arg_roc(self, parser):
        parser.add_argument('-c', '--checkpoint')

    def run_roc(self):
        checkpoint = torch.load(self.args.checkpoint)
        train_args = checkpoint['args']
        bench = self.create_bench_by_args(args)
        bench.restore(checkpoint['model_data'])
        self.draw_roc(bench)

    def run_visualize(self):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        fig.suptitle('violin')
        for i, col in enumerate(['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b']):
            ax = axes[i//4][i%4]
            ax.set_title(col)
            # sns.violinplot(x='sex', y=col, hue='treatment', data=self.df_train, split=True, ax=ax)
            sns.violinplot(y=col, x='treatment', hue='sex', data=self.df_train, split=True, ax=ax)
        plt.show()


t = Table()
t.run()
