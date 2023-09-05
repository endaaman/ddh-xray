import os
from glob import glob
import json
from itertools import combinations

import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics as skmetrics
from scipy import stats
import numpy as np
from pydantic import Field
import lightgbm as lgb

from endaaman.ml import BaseMLCLI, roc_auc_ci

from common import col_target, cols_feature, cols_measure, load_data

# matplotlib.use('TkAgg')


json.encoder.FLOAT_REPR = lambda x: print(x) or format(x, '.8f')

J = os.path.join


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    print(nsamples)
    auc_differences = []
    auc1 = skmetrics.roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = skmetrics.roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = skmetrics.roc_auc_score(y_test.ravel(), p1)
        auc2 = skmetrics.roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)


class CLI(BaseMLCLI):
    class RocArgs(BaseMLCLI.CommonArgs):
        target: str = 'val'

    def run_roc(self, a:RocArgs):
        tt = (
            [
                'Xp images + Measurements',
                # 'out/classification/full_8/resnet50_1/',
                'out/classification/full_8/tf_efficientnet_b8_final',
            ], [
                'Xp images',
                # 'out/classification/full_0/resnet50_final/',
                # 'out/classification/full_0/tf_efficientnet_b8_1//',
                'out/classification/full_0/tf_efficientnet_b8_final',
            ]
        )

        for t in tt:
            name, exp_dir = t
            data = torch.load(J(exp_dir, 'predictions.pt'))
            preds = data['val_preds'].flatten().numpy()
            gts = data['val_gts'].flatten().numpy()
            t.append(preds)
            t.append(gts)
            fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
            auc = skmetrics.auc(fpr, tpr)
            # lower, upper = roc_auc_ci(gts, preds)
            # plt.plot(fpr, tpr, label=f'{name} AUC:{auc:.3f}({lower:.3f}-{upper:.3f})')
            plt.plot(fpr, tpr, label=f'{name} AUC:{auc:.3f}')

        diff, p = permutation_test_between_clfs(tt[0][3], tt[0][2], tt[1][2])

        plt.text(0.58, 0.14, f'permutaion p-value: {p:.5f}', fontstyle='italic')


        plt.ylabel('Sensitivity')
        plt.xlabel('1 - Specificity')
        plt.legend(loc='lower right')
        plt.savefig('out/roc.png')
        plt.show()

    class CenterCropArgs(BaseMLCLI.CommonArgs):
        size: int

    def run_center_crop(self, a:CenterCropArgs):
        d = f'tmp/crop_{a.size}'
        os.makedirs(d, exist_ok=True)
        pp = glob('data/images/*.jpg')
        for p in tqdm(pp):
            name = os.path.split(p)[-1]
            i = Image.open(p)
            w, h = i.size
            x0 = (w - a.size)/2
            y0 = (h - a.size)/2
            x1 = (w + a.size)/2
            y1 = (h + a.size)/2
            i = i.crop((x0, y0, x1, y1))
            i.save(f'{d}/{name}')

    class RocFoldsMeanArgs(BaseMLCLI.CommonArgs):
        noshow: bool = Field(False, cli=('--noshow', ))
        model:str = 'resnet34'
        basedir: str = 'out/classification_resnet'

    def run_roc_folds_mean(self, a):
        matplotlib.use('Agg')
        arm1 = 'integrated'
        arm2 = 'additional'
        # arm2 = 'image'
        result = {
            arm1: [],
            arm2: [],
        }
        for mode in [arm1, arm2]:
            preds = []
            gts = []
            for fold in [1, 2, 3, 4, 5, 6]:
                P = torch.load(
                    J(a.basedir, mode, f'{a.model}_fold{fold}/predictions_last.pt'),
                    map_location=torch.device('cpu'))
                pred = P['val_preds'].flatten()
                gt = P['val_gts'].flatten()

                fpr, tpr, __thresholds = skmetrics.roc_curve(gt, pred)
                auc = skmetrics.auc(fpr, tpr)
                result[mode].append(auc)
                preds.append(pred)
                gts.append(gt)

            preds = torch.cat(preds).numpy()
            gts = torch.cat(gts).numpy()

            fpr, tpr, __thresholds = skmetrics.roc_curve(gts, preds)
            auc = skmetrics.auc(fpr, tpr)
            # lower, upper = roc_auc_ci(gts, preds)
            # plt.plot(fpr, tpr, label=f'{name} AUC:{auc:.3f}({lower:.3f}-{upper:.3f})')
            plt.plot(fpr, tpr, label=f'{mode} AUC:{auc:.3f}')

        tvalue, pvalue = stats.ttest_ind(result[arm1], result[arm2])
        result['tvalue'] = tvalue
        result['pvalue'] = pvalue

        plt.ylabel('Sensitivity')
        plt.xlabel('1 - Specificity')
        plt.legend(loc='lower right')
        plt.savefig(f'data/result/roc_folds_mean_{a.model}.png')
        with open(f'data/result/roc_folds_mean_{a.model}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(result)

        # if not a.noshow:
        #     plt.show()

    def run_p_value(self, a):
        with open('data/result/roc_folds_mean_all.json', mode='r') as f:
            s = json.load(f)

        names = []
        ee = s['experiments']

        for i, e in enumerate(ee):
            names.append(e['name'])
            aa = e['aucs']
            ee[i]['mean'] = np.mean(aa)
            lower, higher = stats.norm.interval(0.95, loc=np.mean(aa), scale=np.sqrt(stats.tvar(aa)/len(aa)))
            ee[i]['ci'] = [lower, higher]

        result = {}
        for (a, b) in combinations(range(len(ee)), 2):
            tvalue, pvalue = stats.ttest_ind(ee[a]['aucs'], ee[b]['aucs'])
            a_name, b_name = ee[a]['name'], ee[b]['name']
            result[f'{a_name}_vs_{b_name}'] = {
                'tvalue': tvalue,
                'pvalue': pvalue,
            }

        s['result'] = result
        with open(f'data/result/roc_folds_mean_all2.json', 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2)

        # plt.ylabel('Sensitivity')
        # plt.xlabel('1 - Specificity')
        # plt.legend(loc='lower right')
        # plt.savefig(f'data/result/roc_folds_mean_{a.model}.png')
        print(result)



    def run_assign_folds_to_table(self, a):
        df = pd.read_excel('data/table.xlsx', index_col=0)

        df['fold'] = -1
        for fold in range(1,7):
            ii = sorted([int(f[:4]) for f in os.listdir(f'data/folds6/fold{fold}/test/images/')])
            df.loc[df.index.isin(ii), 'fold'] = fold
        df.to_excel('data/table2.xlsx')

    def run_gbm_by_folds(self, a):
        # df = pd.read_excel('data/table.xlsx', index_col=0)
        dfs = load_data(0, True, a.seed)
        df = dfs['all']
        for fold in range(1,7):
            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]

            x_train = df_train[cols_measure]
            y_train = df_train[col_target]
            x_valid = df_valid[cols_measure]
            y_valid = df_valid[col_target]

            train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=[])
            # valid_sets = [train_set]
            valid_data = lgb.Dataset(x_valid, label=y_valid, categorical_feature=[])
            valid_sets = [valid_data]

            model = lgb.train(
                params={
                    'objective': 'binary',
                    'num_threads': -1,
                    'max_depth': 3,
                    'bagging_seed': a.seed,
                    'random_state': a.seed,
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

            pred_valid = model.predict(x_valid, num_iteration=model.best_iteration)

            fpr, tpr, thresholds = skmetrics.roc_curve(y_valid, pred_valid)
            auc = skmetrics.auc(fpr, tpr)
            print(fold, auc)





if __name__ == '__main__':
    cli = CLI()
    cli.run()

