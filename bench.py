from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer, KNNImputer

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from datasets import cols_cat, col_target, cols_feature


class Bench:
    def __init__(self, use_fold, seed):
        self.use_fold = use_fold
        self.seed = seed

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

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
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
    def __init__(self, imputer=None, use_optuna=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lgb = opt_lgb if use_optuna else org_lgb
        if imputer:
            self.imputer = {
                'simple': SimpleImputer(missing_values=np.nan, strategy='median'),
                'knn': KNNImputer(n_neighbors=5),
            }[imputer]
        else:
            self.imputer = None


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
            'bagging_seed': self.seed,
            'random_state': self.seed,
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
    def __init__(self, imputer='simple', svm_kernel='rbf', *args, **kwargs):
        super().__init__(*args, **kwargs)
        imputer = imputer or 'simple'
        self.imputer = {
            'simple': SimpleImputer(missing_values=np.nan, strategy='median'),
            'knn': KNNImputer(n_neighbors=5),
        }[imputer]
        self.svm_kernel = svm_kernel

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

class NNModel(nn.Module):
    def __init__(self, num_feature, num_classes):
        super().__init__()
        cfg = [
            64,
            64,
            num_classes,
        ]
        last_feat = num_feature
        layers = []
        for i, n in enumerate(cfg):
            layers.append(nn.Linear(last_feat, n))
            last_feat = n
            if i != len(cfg) - 1:
                layers.append(nn.Dropout())

        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dense(x)
        return x


class NNBench(Bench):
    def __init__(self, batch_size=24, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _train(self, x_train, y_train, x_valid, y_valid):
        # impute
        x_train = x_train.copy()
        y_train = y_train.copy()
        x_train[x_train.isna()] = -1000
        y_train[y_train.isna()] = -1000

        model = NNModel(num_feature=x_train.shape[-1], num_classes=1)

        ds = TensorDataset(
            torch.from_numpy(x_train.values),
            torch.from_numpy(y_train.values))
        loader = DataLoader(ds, batch_size=self.batch_size)
        # for (inputs, labels) in loader:
        #     print(inputs.shape)

        return model

    def _predict(self, model, x):
        # TODO: impl NNModel eval
        # model.eval()
        # print(x)
        # t = model(x)
        # print(t.shape)
        # return t

    def serialize(self):
        pass

    def restore(self, data):
        pass
