import os
import copy
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
import lightgbm as org_lgb
import optuna
import optuna.integration.lightgbm as opt_lgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

from endaaman import Commander


optuna.logging.disable_default_handler()

col_target = 'treatment'
# cols_cat = ['sex', 'breech_presentation']
# cols_val = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

cols_cat = []
cols_val = ['sex', 'breech_presentation', 'left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]
cols_feature = cols_cat + cols_val


class Model:
    def train(self, x, y):
        pass

    def predict(self, x, y):
        pass

    def serialize(self):
        pass

    def restore(self, data):
        pass

class GBMModel():
    def __init__(self):
        pass

    def train(self, x, y):
        pass

    def predict(self, x, y):
        pass

    def serialize(self):
        pass

    def restore(self, data):
        pass


def impute(imp, x):
    return pd.DataFrame(imp.fit(x).transform(x), columns=x.columns)

class Table(Commander):
    def get_suffix(self):
        return '_' + self.args.suffix if self.args.suffix else ''

    def arg_common(self, parser):
        parser.add_argument('-r', '--test-ratio', type=float)
        parser.add_argument('-o', '--use-optuna', action='store_true')
        parser.add_argument('-e', '--seed', type=int, default=42)
        parser.add_argument('-s', '--suffix')

    def pre_common(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='median')

        df = pd.read_excel('data/table.xlsx', index_col=0)
        df_train = df[df['test'] == 0]
        df_test = df[df['test'] == 1]

        df_measure_train = pd.read_excel('data/measurement_train.xlsx', index_col=0)
        df_measure_test = pd.read_excel('data/measurement_test.xlsx', index_col=0)
        df_train = pd.concat([df_train, df_measure_train], axis=1)
        df_test = pd.concat([df_test, df_measure_test], axis=1)

        if self.args.model == 'svm':
            df_train = impute(imp, df_train)
            df_test = impute(imp, df_test)

        self.df_all = pd.concat([df_train, df_test])
        # self.df_all.loc[:, cols_cat]= self.df_all[cols_cat].astype('category')

        if self.args.test_ratio:
            df_train, df_test = train_test_split(self.df_all, test_size=self.args.test_ratio, random_state=self.args.seed)
            # df_train.loc[:, 'test'] = 0
            # df_test.loc[:, 'test'] = 1
        self.df_train = df_train
        self.df_test = df_test

        self.lgb = opt_lgb if self.args.use_optuna else org_lgb

    def run_demo(self):
        print(len(self.df_test))
        print(len(self.df_train))

    def arg_train(self, parser):
        parser.add_argument('--roc', action='store_true')
        parser.add_argument('-m', '--model', default='gbm')

    def run_train(self):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.args.seed)
        # 各foldターゲットのラベルの分布がそろうようにする = stratified K fold
        folds = folds.split(np.arange(len(self.df_train)), y=self.df_train[col_target])
        folds = list(folds)

        preds_valid = np.zeros([len(self.df_train)], np.float32)
        preds_test = np.zeros([5, len(self.df_test)], np.float32)
        df_feature_importance = pd.DataFrame()

        models = []
        for fold in range(5):
            x_train = self.df_train.iloc[folds[fold][0]][cols_feature]
            y_train = self.df_train.iloc[folds[fold][0]][col_target]
            x_valid = self.df_train.iloc[folds[fold][1]][cols_feature]
            y_valid = self.df_train.iloc[folds[fold][1]][col_target]
            x_test = self.df_test[cols_feature]

            print(f'fold: {fold+1}, train: {len(x_train)}, valid: {len(x_valid)}')

            if self.args.model == 'gbm':
                train_data = self.lgb.Dataset(x_train, label=y_train, categorical_feature=cols_cat)
                valid_data = self.lgb.Dataset(x_valid, label=y_valid, categorical_feature=cols_cat)
                gbm_params = {
                    'objective': 'binary', # 目的->2値分類
                    'num_threads': -1,
                    'max_depth': 3,
                    'bagging_seed': self.args.seed,
                    'random_state': self.args.seed,
                    'boosting': 'gbdt',
                    'metric': 'auc',
                    'verbosity': -1,
                }

                model = self.lgb.train(
                    gbm_params, # モデルのパラメータ
                    train_data, # 学習データ
                    1000, # 学習を繰り返す最大epoch数, epoch = モデルの学習回数
                    valid_sets=[train_data, valid_data], # 検証データ
                    verbose_eval=200, # 100 epoch ごとに経過を表示する
                    early_stopping_rounds=150, # 150epoch続けて検証データのロスが減らなかったら学習を中断する
                    categorical_feature=cols_cat,
                )
                tmp = pd.DataFrame()
                tmp['feature'] = cols_feature
                tmp['importance'] = model.feature_importance()
                tmp['fold'] = fold + 1
                df_feature_importance = pd.concat([df_feature_importance, tmp], axis=0)

                preds_valid[folds[fold][1]] = model.predict(x_valid, num_iteration=model.best_iteration)
                preds_test[fold] = model.predict(x_test, num_iteration=model.best_iteration)
            elif self.args.model == 'svm':
                model = SVC(kernel='linear', random_state=None)
                model.fit(x_train, y_train)
                pred_train = model.predict(x_train)
                # accuracy_train = metrics.auc(y_train, pred_train)

                preds_valid[folds[fold][1]] = model.predict(x_valid)
                preds_test[fold] = model.predict(x_test)
            else:
                raise Exception(f'Invalid model: {self.args.model}')

            models.append(model)

        score = metrics.roc_auc_score(self.df_train[col_target], preds_valid)
        print(f'CV AUC: {score:.6f}')

        if self.args.model == 'gbm':
            df_tmp = df_feature_importance.groupby('feature').agg('mean').reset_index()
            df_tmp = df_tmp.sort_values('importance', ascending=False)
            print(df_tmp[['feature', 'importance']])
            # df_tmp.to_csv('out/importance.csv')
            model_data = [m.model_to_string() for m in models]
        else:
            model_data = models

        # p = f'out/model{self.get_suffix()}.txt'
        # model.save_model(p)
        checkpoint = {
            'model_data': model_data,
        }
        p = f'out/model{self.get_suffix()}.pth'
        torch.save(checkpoint, p)
        print(f'wrote {p}')

        if self.args.roc:
            self.draw_roc(models)

    def draw_roc(self, models):
        x_train = self.df_train[cols_feature]
        y_train = self.df_train[col_target]

        x_test = self.df_test[cols_feature]
        y_test = self.df_test[col_target]

        preds_train = []
        preds_test = []
        for model in models:
            if self.args.model == 'gbm':
                preds_train.append(model.predict(x_train, num_iteration=model.best_iteration))
                preds_test.append(model.predict(x_test, num_iteration=model.best_iteration))
            elif self.args.model == 'svm':
                preds_train.append(model.predict(x_train))
                preds_test.append(model.predict(x_test))
            else:
                raise Exception(f'Invalid model: {self.args.model}')
        pred_train = np.mean(preds_train, axis=0)
        pred_test = np.mean(preds_test, axis=0)

        # model = models[0]
        # pred_train = model.predict(x_train, num_iteration=model.best_iteration)
        # pred_test = model.predict(x_test, num_iteration=model.best_iteration)

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
        parser.add_argument('-w', '--weights', default='out/model.txt')

    def run_roc(self):
        model = self.lgb.Booster(model_file=self.args.weight)
        self.draw_roc(model)


Table().run()
