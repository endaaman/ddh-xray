import random
import os
import copy
from collections import OrderedDict
from dataclasses import dataclass

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
import seaborn as sns
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import scipy.stats as st
from pydantic import Field
import lightgbm as lgb

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

from endaaman.ml import BaseMLCLI, get_global_seed
from common import load_data, col_target, cols_feature, cols_measure

J = os.path.join

def inv_sigmoid(x):
    return 1/(1+np.exp(-x))

def calc_metrics(gt, pred):
    fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
    auc = skmetrics.auc(fpr, tpr)
    ci = auc_ci(gt, pred)

    ii = {}
    f1_scores = [skmetrics.f1_score(gt, pred > t) for t in thresholds]
    acc_scores = [skmetrics.accuracy_score(gt, pred > t) for t in thresholds]

    ii['f1'] = np.argmax(f1_scores)
    ii['acc'] = np.argmax(acc_scores)
    ii['youden'] = np.argmax(tpr - fpr)
    ii['top-left'] = np.argmin((- tpr + 1) ** 2 + fpr ** 2)

    scores = pd.DataFrame({
        k: {
            'acc': acc_scores[i],
            'f1': f1_scores[i],
            'recall': tpr[i],
            'specificity': -fpr[i]+1,
            'thres': thresholds[i],
        } for k, i in ii.items()
    }).transpose()
    return Metrics(fpr, tpr, thresholds, auc, ci, scores)


@dataclass
class Result:
    gt: np.ndarray
    pred: np.ndarray

@dataclass
class Metrics:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    ci: np.ndarray
    scores: pd.DataFrame

    @classmethod
    def from_result(cls, r):
        return calc_metrics(r.gt, r.pred)



def train_single_gbm(x_train, y_train, x_valid, y_valid, seed):
    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=[])
    valid_sets = [train_set]
    if np.any(x_valid):
        valid_data = lgb.Dataset(x_valid, label=y_valid, categorical_feature=[])
        valid_sets += [valid_data]

    model = lgb.train(
        params={
            'objective': 'binary',
            'num_threads': -1,
            'max_depth': 3,
            'bagging_seed': seed,
            'random_state': seed,
            'boosting': 'gbdt',
            'metric': 'auc',
            'verbosity': -1,
            'zero_as_missing': True,
        },
        train_set=train_set,
        num_boost_round=10000,
        valid_sets=valid_sets,
        # early_stopping_rounds=150,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(False)
        ],
        categorical_feature=[],
    )
    return model


def train_gbm(df, target_col, num_folds=5, reduction='median', seed=get_global_seed()):
    df_train = df[df['test'] < 1].drop(['test'], axis=1)
    df_test = df[df['test'] > 0].drop(['test'], axis=1)
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = folds.split(np.arange(len(df_train)), y=df_train[target_col])
    folds = list(folds)
    models = []

    importances = []
    for fold in tqdm(range(num_folds), leave=False):
        # print(f'fold {fold+1}/{num_folds}')
        df_x = df_train.drop([target_col], axis=1)
        df_y =  df_train[target_col]
        vv = [
            df_x.iloc[folds[fold][0]].values, # x_train
            df_y.iloc[folds[fold][0]].values, # y_train
            df_x.iloc[folds[fold][1]].values, # x_valid
            df_y.iloc[folds[fold][1]].values, # y_valid
        ]
        vv = [v.copy() for v in vv]
        model = train_single_gbm(*vv, seed)
        models.append(model)

        importances.append(model.feature_importance(importance_type='gain'))

    cols = list(set(df.columns) - {'treatment', 'test'})
    importance = pd.DataFrame(columns=cols, data=importances)
    mean = importance.mean(axis=0)
    importance = importance.transpose()
    importance['mean'] = mean
    importance = importance.sort_values(by='mean', ascending=False)
    importance = importance[importance.columns[[-1, *range(num_folds)]]]

    preds = []
    for model in models:
        x = df_test.drop([target_col], axis=1).values
        pred = model.predict(x, num_iteration=model.best_iteration)
        preds.append(pred)

    match reduction:
        case 'mean':
            pred = np.mean(preds, axis=0)
        case 'median':
            pred = np.median(preds, axis=0)
        case _:
            raise RuntimeError(f'Invalid reduction: {reduction}')

    gt =  df_test[target_col].values
    return Result(gt, pred), importance

def auc_ci(y_true, y_score):
    y_true = y_true.astype(bool)
    AUC = skmetrics.roc_auc_score(y_true, y_score)
    N1 = sum(y_true)
    N2 = sum(~y_true)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = np.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    return np.clip([lower, upper], 0.0, 1.0)

def calc_metrics(gt, pred):
    fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
    auc = skmetrics.auc(fpr, tpr)
    ci = auc_ci(gt, pred)

    ii = {}
    f1_scores = [skmetrics.f1_score(gt, pred > t) for t in thresholds]
    acc_scores = [skmetrics.accuracy_score(gt, pred > t) for t in thresholds]

    ii['f1'] = np.argmax(f1_scores)
    ii['acc'] = np.argmax(acc_scores)
    ii['youden'] = np.argmax(tpr - fpr)
    ii['top-left'] = np.argmin((- tpr + 1) ** 2 + fpr ** 2)

    scores = pd.DataFrame({
        k: {
            'acc': acc_scores[i],
            'f1': f1_scores[i],
            'recall': tpr[i],
            'specificity': -fpr[i]+1,
            'thres': thresholds[i],
        } for k, i in ii.items()
    }).transpose()
    return Metrics(fpr, tpr, thresholds, auc, ci, scores)


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class GbmArgs(CommonArgs):
        with_cnn_p: bool = Field(False, cli=('--with-cnn-p', ))

    def run_gbm(self, a:GbmArgs):
        dfs = load_data(test_ratio=-1, normalize_features=True, seed=a.seed)

        for k, df in dfs.items():
            cc = cols_measure + [col_target]
            dfs[k] = df[cc]
        df_train = dfs['train']
        df_test = dfs['test']
        df_train['test'] = 0
        df_test['test'] = 1

        if a.with_cnn_p:
            basedir = 'out/classification/full_0/tf_efficientnet_b0'
            train_preds = np.load(J(basedir, 'train_preds.npy'))
            test_preds = np.load(J(basedir, 'val_preds.npy'))
            col_additional = 'CNN prediction'
            # df_train[col_additional] = inv_sigmoid(train_preds)
            # df_test[col_additional] = inv_sigmoid(test_preds)
            df_train[col_additional] = train_preds
            df_test[col_additional] = test_preds
            print('add')
        df = pd.concat([df_train, df_test])
        print(df)
        print(df.columns)

        result, importance = train_gbm(df=df, target_col=col_target, seed=a.seed)
        # print(result)
        print(importance)
        p = 'out/importance_with_cnn_p.xlsx' if a.with_cnn_p else 'out/importance.xlsx'
        importance.to_excel(p)

        m:Metrics = Metrics.from_result(result)
        print(m)
        print(m.auc)







if __name__ == '__main__':
    cli = CLI()
    cli.run()
