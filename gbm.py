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
import lightgbm as lgb

from endaaman import Commander


col_target = 'treatment'
# cols_feature = df.columns.values.tolist()[1:-1] # exclude label, treatment
cols_feature = ['sex', 'family_history', 'breech_presentation', 'skin_laterality', ' limb_limitation']
cols_cat = copy.deepcopy(cols_feature)

class GBM(Commander):
    def get_data(self):
        df = pd.read_excel('data/table.xlsx')
        df_train = df[df['test'] == 0]
        df_test = df[df['test'] == 1]
        return df, df_train, df_test

    def run_train(self):
        df, df_train, df_test = self.get_data()

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = folds.split(np.arange(len(df_train)), y=df_train[col_target]) # 各foldターゲットのラベルの分布がそろうようにする = stratified K fold
        folds = list(folds)

        params = {
            'objective': 'binary', # 目的->2値分類
            'num_threads': -1,
            'bagging_seed': 42, # random seed の固定
            'random_state': 42, # random seed の固定
            'boosting': 'gbdt',
            'metric': 'auc', # 評価変数->AUC
            'verbosity': -1,
        }

        preds_valid = np.zeros([len(df_train)], np.float32)
        preds_test = np.zeros([5, len(df_test)], np.float32)
        df_feature_importance = pd.DataFrame()

        for fold in range(5):
            x_train = df_train.iloc[folds[fold][0]][cols_feature]
            y_train = df_train.iloc[folds[fold][0]][col_target]
            x_valid = df_train.iloc[folds[fold][1]][cols_feature]
            y_valid = df_train.iloc[folds[fold][1]][col_target]
            x_test = df_test[cols_feature]

            print(f'fold: {fold+1}, train: {len(x_train)}, valid: {len(x_valid)}')
            train_data = lgb.Dataset(x_train, label=y_train, categorical_feature=cols_cat)
            valid_data = lgb.Dataset(x_valid, label=y_valid, categorical_feature=cols_cat)

            model = lgb.train(
                params, # モデルのパラメータ
                train_data, # 学習データ
                1000, # 学習を繰り返す最大epoch数, epoch = モデルの学習回数
                valid_sets=[train_data, valid_data], # 検証データ
                verbose_eval=100, # 100 epoch ごとに経過を表示する
                early_stopping_rounds=150, # 150epoch続けて検証データのロスが減らなかったら学習を中断する
            )

            preds_valid[folds[fold][1]] = model.predict(x_valid, num_iteration=model.best_iteration) # 検証データに対する予測を実行
            preds_test[fold] = model.predict(x_test, num_iteration=model.best_iteration)  # テストデータに対する予測を実行

            # 特徴量の重要度を記録
            tmp = pd.DataFrame()
            tmp['feature'] = cols_feature
            tmp['importance'] = model.feature_importance()
            tmp['fold'] = fold + 1
            df_feature_importance = pd.concat([df_feature_importance, tmp], axis=0)


        score = metrics.roc_auc_score(df_train[col_target], preds_valid)
        print(f'CV AUC: {score:.6f}')

        df_tmp = df_feature_importance.groupby('feature').agg('mean').reset_index()
        df_tmp = df_tmp.sort_values('importance', ascending=False)
        print(df_tmp[['feature', 'importance']])
        df_tmp.to_csv('out/importance.csv')

        p = 'out/model.txt'
        model.save_model(p)
        print(f'wrote {p}')

    def run_roc(self):
        df, df_train, df_test = self.get_data()

        model = lgb.Booster(model_file='out/model.txt')

        x_train = df_train[cols_feature]
        y_train = df_train[col_target]

        x_test = df_test[cols_feature]
        y_test = df_test[col_target]

        pred_train = model.predict(x_train, num_iteration=model.best_iteration)
        pred_test = model.predict(x_test, num_iteration=model.best_iteration)

        for (t, y, pred) in (('train', y_train, pred_train), ('test', y_test, pred_test)):
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{t} auc = {auc:.2f})')

        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        p = 'out/roc.png'
        plt.savefig(p)
        plt.show()
        print(f'wrote {p}')



GBM().run()
