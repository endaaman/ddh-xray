from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer, KNNImputer
import lightgbm as lgb
# import optuna.integration.lightgbm as opt_lgb
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from endaaman import Timer
from common import cols_feature

class Bench:
    def __init__(self, num_folds, seed):
        self.num_folds = num_folds
        self.seed = seed
        self.training_time = -1
        self.predicting_time = -1

    def preprocess(self, x, y=None):
        return x, y

    def train(self, df_train, target_col):
        t = Timer()
        t.start()
        folds = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        folds = folds.split(np.arange(len(df_train)), y=df_train[target_col])
        folds = list(folds)
        models = []
        for fold in range(self.num_folds):
            print(f'fold {fold+1}/{self.num_folds}')
            df_x = df_train.drop([target_col], axis=1)
            df_y =  df_train[target_col]
            vv = [
                df_x.iloc[folds[fold][0]], # x_train
                df_y.iloc[folds[fold][0]], # y_train
                df_x.iloc[folds[fold][1]], # x_valid
                df_y.iloc[folds[fold][1]], # y_valid
            ]
            vv = [v.copy() for v in vv]
            vv[0], vv[1] = self.preprocess(*vv[:2])
            vv[2], vv[3] = self.preprocess(*vv[2:])
            model = self._train(*vv, fold)
            models.append(model)
        t.end()
        self.training_time = t.ms()
        self.models = models

    def _train(self, x_train, y_train, x_valid, y_valid, fold):
        pass

    def predict(self, x):
        t = Timer()
        t.start()
        x, _ = self.preprocess(x.copy())
        preds = []
        for model in self.models:
            preds.append(self._predict(model, x.copy()))
        t.end()
        self.predicting_time = t.ms()
        return np.stack(preds, axis=1)

    def _predict(self, model, x):
        pass

    def serialize(self):
        pass

    def restore(self, data):
        pass


class LightGBMBench(Bench):
    def __init__(self, imputer=None, use_optuna=False, **kwargs):
        super().__init__(**kwargs)
        self.df_feature_importance = pd.DataFrame()

    def preprocess(self, x, y=None):
        # label smoothing
        return x, y

    def train(self, df_train, target_col):
        super().train(df_train, target_col)

        df_tmp = self.df_feature_importance.groupby('feature').agg('mean').reset_index()
        df_tmp = df_tmp.sort_values('importance', ascending=False)

        print('IMPORTANCE')
        print(df_tmp[['feature', 'importance']])
        # df_tmp.to_excel('out/imp_split.xlsx')
        df_tmp.to_excel('out/imp_gain.xlsx')

    def _train(self, x_train, y_train, x_valid, y_valid, fold):
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
                'bagging_seed': self.seed,
                'random_state': self.seed,
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
        # model = self.lgb.train(
        #     gbm_params, # モデルのパラメータ
        #     train_data, # 学習データ
        #     1000, # 学習を繰り返す最大epoch数, epoch = モデルの学習回数
        #     valid_sets=valid_sets,
        #     verbose_eval=200,
        #     early_stopping_rounds=150,
        #     categorical_feature=[],
        # )

        tmp = pd.DataFrame()
        tmp['feature'] = cols_feature
        tmp['importance'] = model.feature_importance(importance_type='gain')
        # tmp['importance'] = model.feature_importance(importance_type='split')
        tmp['fold'] = fold + 1
        self.df_feature_importance = pd.concat([self.df_feature_importance, tmp], axis=0)

        return model

    def _predict(self, model, x):
        y = model.predict(x, num_iteration=model.best_iteration)
        return y

    def serialize(self):
        return [m.model_to_string() for m in self.models]

    def restore(self, data):
        self.models = []
        for m in data:
            self.models.append(self.lgb.Booster(model_str=m))


class SVMBench(Bench):
    def __init__(self, imputer='simple', svm_kernel='rbf', **kwargs):
        super().__init__(**kwargs)
        imputer = imputer or 'simple'
        self.imputer = {
            'simple': SimpleImputer(missing_values=np.nan, strategy='median'),
            'knn': KNNImputer(n_neighbors=5),
            'none': None,
        }[imputer]
        self.svm_kernel = svm_kernel

    def preprocess(self, x, y=None):
        if self.imputer:
            self.imputer.fit(x)
            x = self.imputer.transform(x)
        x[np.isnan(x)] = -1
        return x, y

    def _train(self, x_train, y_train, x_valid, y_valid, fold):
        param_list = [0.001, 0.01, 0.1, 1, 10]
        best_score = 0
        best_parameters = {}
        best_model = None
        for gamma in tqdm(param_list):
            t = tqdm(param_list, leave=False)
            for C in t:
                # model = SVC(kernel=self.svm_kernel, gamma=gamma, C=C, random_state=None)
                model = SVR(kernel=self.svm_kernel, gamma=gamma, C=C)
                model.fit(x_train, y_train)
                score = model.score(x_valid, y_valid)
                t.set_description(f'score: {score:.3f} best: {best_score:.3f}')
                t.refresh()
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
        y = model.predict(x)
        return y

    def serialize(self):
        pass

    def restore(self, data):
        pass

class Dense(nn.Module):
    def __init__(self, a, b=None):
        super().__init__()
        if not b:
            b = a
        self.dense = nn.Linear(a, b)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
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
                layers += [
                    # nn.BatchNorm1d(n),
                    nn.Dropout(p=0.2),
                    nn.ReLU(inplace=True),
                ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)


class NNBench(Bench):
    def __init__(self, epoch=1000, batch_size=48, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device

    def as_loader(self, x, y):
        ds = TensorDataset(
            torch.from_numpy(x.values).type(torch.FloatTensor),
            torch.from_numpy(y.values).type(torch.FloatTensor)[:, None])
        return DataLoader(ds, batch_size=self.batch_size)

    def preprocess(self, x, y=None):
        # label smoothing
        if np.any(y):
            q = x.isnull().any(axis=1)
            epsilon = 0.6
            y[q] *= epsilon
            y[q] += (1-epsilon)/2
        x[x.isna()] = -1
        return x, y

    def _train(self, x_train, y_train, x_valid, y_valid, fold):
        model = NNModel(num_feature=x_train.shape[-1], num_classes=1)
        train_loader = self.as_loader(x_train, y_train)
        valid_loader = self.as_loader(x_valid, y_valid)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion = nn.BCELoss()

        e = tqdm(range(self.epoch))
        for epoch in e:
            train_losses = []
            model.train()
            for (x, y) in train_loader:
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            model.eval()

            val_losses = []
            with torch.no_grad():
                for (x, y) in valid_loader:
                    z = model(x)
                    loss = criterion(z, y)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            if lr < 0.0000001:
                break
            e.set_description(f'[loss] train: {train_loss:.2f} val: {val_loss:.2f} lr: {lr:.7f}')
            e.refresh()

        return model

    def _predict(self, model, x):
        # x[x.isna()] = -1000
        outputs = []
        for start in tqdm(range(0, len(x), self.batch_size)):
            batch = x[start:start + self.batch_size]
            input_tensor = torch.from_numpy(batch.values).type(torch.FloatTensor)
            output_tensor = model(input_tensor)
            outputs.append(output_tensor)
        outputs = torch.cat(outputs)
        y = outputs.cpu().detach().numpy().squeeze(-1)
        return y

    def serialize(self):
        pass

    def restore(self, data):
        pass

if __name__ == '__main__':
    m = NNModel(10, 1)
    i = torch.ones([24, 10])
    print(m(i).shape)
