from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import optuna.integration.lightgbm as opt_lgb
import lightgbm as org_lgb
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from datasets import cols_cat, col_target, cols_feature


class Bench:
    def __init__(self, use_fold, seed):
        self.use_fold = use_fold
        self.seed = seed

    def impute(self, x):
        # return pd.DataFrame(self.imp.fit(x).transform(x), columns=x.columns)
        return self.imputer.fit(x).transform(x)

    def preprocess(self, x, y):
        return x, y

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
            print(f'fold {fold}/5')
            vv = [
                df_train.iloc[folds[fold][0]][cols_feature], # x_train
                df_train.iloc[folds[fold][0]][col_target],   # y_train
                df_train.iloc[folds[fold][1]][cols_feature], # x_valid
                df_train.iloc[folds[fold][1]][col_target],   # y_valid
            ]
            vv = [v.copy() for v in vv]
            vv[0], vv[1] = self.preprocess(*vv[:2])
            vv[2], vv[3]  = self.preprocess(*vv[2:])
            model = self._train(*vv)
            models.append(model)
        self.models = models

    def _train(self, x_train, y_train, x_valid, y_valid):
        pass

    def predict(self, x):
        preds = []
        for model in self.models:
            preds.append(self._predict(model, x.copy()))
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

    def preprocess(self, x, y):
        # label smoothing
        q = x.isnull().any(axis=1)
        epsilon = 0.4
        y[q] *= epsilon
        y[q] += (1-epsilon)/2
        if self.imputer:
            x = self.impute(x)
        return x, y

    def _train(self, x_train, y_train, x_valid, y_valid):
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

class SkipLinear(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.dense = nn.Linear(a, a)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.dense(x)
        x = x + y
        x = self.dropout(x)
        x = self.relu(x)
        return x


class NNModel(nn.Module):
    def __init__(self, num_feature, num_classes):
        super().__init__()
        cfg = [
            64,
            128,
            128,
            64,
            num_classes,
        ]
        last_feat = num_feature
        layers = []
        for i, n in enumerate(cfg):
            layers.append(nn.Linear(last_feat, n))
            last_feat = n

            if i != len(cfg) - 1:
                layers.append(nn.Dropout(p=0.2))
                layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)


class NNBench(Bench):
    def __init__(self, epoch=2000, batch_size=24, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device

    def as_loader(self, x, y):
        ds = TensorDataset(
            torch.from_numpy(x.values).type(torch.FloatTensor),
            torch.from_numpy(y.values).type(torch.FloatTensor)[:, None])
        return DataLoader(ds, batch_size=self.batch_size)

    def preprocess(self, x, y):
        # label smoothing
        q = x.isnull().any(axis=1)
        epsilon = 0.4
        y[q] *= epsilon
        y[q] += (1-epsilon)/2

        x[x.isna()] = -1000
        y[y.isna()] = -1000
        return x, y

    def _train(self, x_train, y_train, x_valid, y_valid):
        model = NNModel(num_feature=x_train.shape[-1], num_classes=1)
        train_loader = self.as_loader(x_train, y_train)
        valid_loader = self.as_loader(x_valid, y_valid)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        criterion = nn.BCELoss()

        e = tqdm(range(self.epoch))
        for epoch in e:
            losses = []
            model.train()
            for (x, y) in train_loader:
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = np.mean(losses)
            scheduler.step(train_loss)
            model.eval()
            losses = []
            with torch.no_grad():
                for (x, y) in valid_loader:
                    z = model(x)
                    loss = criterion(z, y)
                    losses.append(loss.item())
            valid_loss = np.mean(losses)

            lr = optimizer.param_groups[0]['lr']
            if lr < 0.0000001:
                break
            e.set_description(f'[loss] train: {train_loss:.2f} val: {valid_loss:.2f} lr: {lr:.7f}')
            e.refresh()

        return model

    def _predict(self, model, x):
        x[x.isna()] = -1000
        outputs = []
        for start in tqdm(range(0, len(x), self.batch_size)):
            batch = x[start:start + self.batch_size]
            input_tensor = torch.from_numpy(batch.values).type(torch.FloatTensor)
            output_tensor = model(input_tensor)
            outputs.append(output_tensor)
        outputs = torch.cat(outputs)
        return outputs.cpu().detach().numpy()

    def serialize(self):
        pass

    def restore(self, data):
        pass

if __name__ == '__main__':
    m = NNModel(10, 1)
    i = torch.ones([24, 10])
    print(m(i).shape)
