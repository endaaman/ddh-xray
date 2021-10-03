import os
import copy
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
import lightgbm as org_lgb
import optuna
import optuna.integration.lightgbm as opt_lgb
from sklearn.model_selection import train_test_split

from endaaman import Commander
from bench import SVMBench, LightGBMBench
from datasets import cols_cat, col_target, cols_feature


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
        df = pd.read_excel('data/table.xlsx', index_col=0)
        df_train = df[df['test'] == 0]
        df_test = df[df['test'] == 1]

        df_measure_train = pd.read_excel('data/measurement_train.xlsx', index_col=0)
        df_measure_test = pd.read_excel('data/measurement_test.xlsx', index_col=0)
        df_train = pd.concat([df_train, df_measure_train], axis=1)
        df_test = pd.concat([df_test, df_measure_test], axis=1)

        self.df_all = pd.concat([df_train, df_test])
        # self.df_all.loc[:, cols_cat]= self.df_all[cols_cat].astype('category')

        if self.args.test_ratio:
            df_train, df_test = train_test_split(self.df_all, test_size=self.args.test_ratio, random_state=self.args.seed)
            # df_train.loc[:, 'test'] = 0
            # df_test.loc[:, 'test'] = 1
        self.df_train = df_train
        self.df_test = df_test

    def run_demo(self):
        print(len(self.df_test))
        print(len(self.df_train))

    def create_bench_by_args(self, args):
        return {
            'gbm': lambda: LightGBMBench(
                use_fold=not args.no_fold,
                seed=args.seed,
                imputer=args.imputer,
                use_optuna=args.optuna),
            'svm': lambda: SVMBench(
                use_fold=not args.no_fold,
                seed=args.seed,
                imputer=args.imputer,
                svm_kernel=args.kernel,
            ),
        }[args.model]()

    def arg_train(self, parser):
        parser.add_argument('--roc', action='store_true')
        parser.add_argument('--no-fold', action='store_true')
        parser.add_argument('-i', '--imputer')
        parser.add_argument('-m', '--model', default='gbm', choices=['gbm', 'svm'])
        parser.add_argument('-k', '--kernel', default='rbf')

    def run_train(self):
        bench = self.create_bench_by_args(self.args)

        preds_valid = np.zeros([len(self.df_train)], np.float32)
        preds_test = np.zeros([5, len(self.df_test)], np.float32)
        df_feature_importance = pd.DataFrame()

        bench.train(self.df_train)
        score = metrics.roc_auc_score(self.df_train[col_target], preds_valid)
        print(f'CV AUC: {score:.6f}')

        # p = f'out/model{self.get_suffix()}.txt'
        # model.save_model(p)
        checkpoint = {
            'args': self.args,
            'model_data': bench.serialize(),
        }
        p = f'out/model{self.get_suffix()}.pth'
        torch.save(checkpoint, p)
        print(f'wrote {p}')

        if self.args.roc:
            self.draw_roc(bench)

    def draw_roc(self, bench):
        x_train = self.df_train[cols_feature]
        y_train = self.df_train[col_target]

        x_test = self.df_test[cols_feature]
        y_test = self.df_test[col_target]

        pred_train = bench.predict(x_train)
        pred_test = bench.predict(x_test)

        for (t, y, pred) in (('train', y_train, pred_train), ('test', y_test, pred_test)):
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{t} auc = {auc:.2f})')

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


Table().run()
