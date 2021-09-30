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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

from endaaman import Commander


SEED = 42

optuna.logging.disable_default_handler()

col_target = 'treatment'
# cols_cat = ['sex', 'breech_presentation']
# cols_val = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

cols_cat = []
cols_val = ['sex', 'breech_presentation', 'left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]
cols_feature = cols_cat + cols_val


class Bench:
    def __init__(self, use_fold, use_optuna, imputer=None):
        self.use_fold = use_fold
        self.lgb = opt_lgb if use_optuna else org_lgb

        if imputer:
            self.imputer = {
                'simple': SimpleImputer(missing_values=np.nan, strategy='median'),
                'knn': KNNImputer(n_neighbors=5),
            }[imputer]
        else:
            self.imputer = None
        self.svm_kernel = 'rbf'

    def impute(self, x):
        # return pd.DataFrame(self.imp.fit(x).transform(x), columns=x.columns)
        return self.imputer.fit(x).transform(x)

    def train(self, df_train):
        if not self.use_fold:
            x_train = df_train[cols_feature]
            y_train = df_train[col_target]
            x_valid = np.array([[]])
            y_valid = np.array([])
            model = self._train(x_train, y_train, x_valid, y_valid)
            self.models = [model]
            return

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        # 各foldターゲットのラベルの分布がそろうようにする = stratified K fold
        folds = folds.split(np.arange(len(df_train)), y=df_train[col_target])
        folds = list(folds)
        models = []
        for fold in range(5):
            x_train = df_train.iloc[folds[fold][0]][cols_feature]
            y_train = df_train.iloc[folds[fold][0]][col_target]
            x_valid = df_train.iloc[folds[fold][1]][cols_feature]
            y_valid = df_train.iloc[folds[fold][1]][col_target]
            model = self._train(x_train, y_train, x_valid, y_valid)
            models.append(model)
        self.models = models

    def _train(self, x_train, y_train, x_valid, y_valid):
        pass

    def predict(self, x):
        preds = []
        for model in self.models:
            preds.append(self._predict(model, x))
        return np.mean(preds, axis=0)

    def _predict(self, model, x):
        pass

    def serialize(self):
        pass

    def restore(self, data):
        pass

class LightGBMBench(Bench):
    def train(self, df_train):
        super().train(df_train)
        # df_tmp = df_feature_importance.groupby('feature').agg('mean').reset_index()
        # df_tmp = df_tmp.sort_values('importance', ascending=False)
        # print(df_tmp[['feature', 'importance']])
        # # df_tmp.to_csv('out/importance.csv')

    def _train(self, x_train, y_train, x_valid, y_valid):
        if self.imputer:
            print('use imputer')
            x_train = self.impute(x_train)
            x_valid = self.impute(x_valid)
        gbm_params = {
            'objective': 'binary', # 目的->2値分類
            'num_threads': -1,
            'max_depth': 3,
            'bagging_seed': SEED,
            'random_state': SEED,
            'boosting': 'gbdt',
            'metric': 'auc',
            'verbosity': -1,
        }

        train_data = self.lgb.Dataset(x_train, label=y_train, categorical_feature=cols_cat)
        valid_sets = [train_data]
        if np.any(x_valid):
            valid_data = self.lgb.Dataset(x_valid, label=y_valid, categorical_feature=cols_cat)
            valid_sets += [valid_data]

        model = self.lgb.train(
            gbm_params, # モデルのパラメータ
            train_data, # 学習データ
            1000, # 学習を繰り返す最大epoch数, epoch = モデルの学習回数
            valid_sets=valid_sets,
            verbose_eval=200, # 100 epoch ごとに経過を表示する
            early_stopping_rounds=150, # 150epoch続けて検証データのロスが減らなかったら学習を中断する
            categorical_feature=cols_cat,
        )
        # tmp = pd.DataFrame()
        # tmp['feature'] = cols_feature
        # tmp['importance'] = model.feature_importance()
        # tmp['fold'] = fold + 1
        # df_feature_importance = pd.concat([df_feature_importance, tmp], axis=0)
        #
        # preds_valid[folds[fold][1]] = model.predict(x_valid, num_iteration=model.best_iteration)
        # preds_test[fold] = model.predict(x_test, num_iteration=model.best_iteration)
        return model

    def _predict(self, model, x):
        return model.predict(x, num_iteration=model.best_iteration)

    def serialize(self):
        return [m.model_to_string() for m in self.models]

    def restore(self, data):
        self.models = []
        for m in data:
            self.models.append(self.lgb.Booster(model_str=m))

class SVMBench(Bench):
    def _train(self, x_train, y_train, x_valid, y_valid):
        x_train = self.impute(x_train)
        x_valid = self.impute(x_valid)

        param_list = [0.001, 0.01, 0.1, 1, 10]
        best_score = 0
        best_parameters = {}
        best_model = None
        for gamma in tqdm(param_list):
            for C in tqdm(param_list, leave=False):
                model = SVC(kernel=self.svm_kernel, gamma=gamma, C=C, random_state=None)
                model.fit(x_train, y_train)
                score = model.score(x_valid, y_valid)
                if score > best_score:
                    best_score = score
                    best_parameters = {'gamma' : gamma, 'C' : C}
                    best_model = model

        # model = SVC(kernel=self.svm_kernel, random_state=None, **best_parameters)
        # pred_train = model.predict(x_train)
        # accuracy_train = metrics.auc(y_train, pred_train)
        # preds_valid[folds[fold][1]] = model.predict(x_valid)
        # preds_test[fold] = model.predict(x_test)
        return best_model

    def _predict(self, model, x):
        x = self.impute(x)
        return model.predict(x)

    def serialize(self):
        pass

    def restore(self, data):
        pass

bench_table = {
    'gbm': LightGBMBench,
    'svm': SVMBench,
}


class Table(Commander):
    def get_suffix(self):
        return '_' + self.args.suffix if self.args.suffix else ''

    def arg_common(self, parser):
        parser.add_argument('-r', '--test-ratio', type=float)
        parser.add_argument('-e', '--seed', type=int, default=42)
        parser.add_argument('-s', '--suffix')
        parser.add_argument('-o', '--optuna', action='store_true')

    def pre_common(self):
        global SEED
        SEED = self.args.seed
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
            df_train, df_test = train_test_split(self.df_all, test_size=self.args.test_ratio, random_state=SEED)
            # df_train.loc[:, 'test'] = 0
            # df_test.loc[:, 'test'] = 1
        self.df_train = df_train
        self.df_test = df_test

    def run_demo(self):
        print(len(self.df_test))
        print(len(self.df_train))

    def create_bench_by_args(self, args):
        return bench_table[args.model](
            use_fold=not args.no_fold,
            use_optuna=args.optuna,
            imputer=args.imputer)

    def arg_train(self, parser):
        parser.add_argument('--roc', action='store_true')
        parser.add_argument('--no-fold', action='store_true')
        parser.add_argument('-i', '--imputer')
        parser.add_argument('-m', '--model', default='gbm', choices=['gbm', 'svm'])

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
