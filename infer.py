import os
import math
from collections import namedtuple

import fire
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn import metrics

from common import *


class Visualize():
    def roc(self):
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


    def feature(self):
        df_tmp = df_feature_importance.groupby('feature').agg('mean').reset_index()
        df_tmp = df_tmp.sort_values('importance', ascending=False)
        print(df_tmp[['feature', 'importance']])


if __name__ == '__main__':
    fire.Fire(Visualize)
