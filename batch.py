import os
from glob import glob
import json
from itertools import combinations

import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as mp3d
from sklearn import metrics as skmetrics
from scipy import stats
import numpy as np
from pydantic import Field
import lightgbm as lgb

from endaaman.ml import BaseMLCLI, roc_auc_ci

from datasets import read_label_as_df
from common import col_target, cols_feature, cols_measure, load_data


# matplotlib.use('TkAgg')


sns.set_style('white')
sns.set_palette('tab10')
# default_palette = sns.color_palette()
# brightened_palette = sns.color_palette("pastel", n_colors=len(default_palette) - 2)
# sns.set_palette(brightened_palette)

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。

# plt.rcParams['xtick.direction'] = 'in' #x軸の目盛りの向き
# plt.rcParams["xtick.minor.visible"] = True  #x軸補助目盛りの追加
# plt.rcParams['xtick.bottom'] = True  #x軸の上部目盛り
plt.rcParams['ytick.direction'] = 'in' #y軸の目盛りの向き
plt.rcParams["ytick.minor.visible"] = True  #y軸補助目盛りの追加
plt.rcParams['ytick.left'] = True  #y軸の右部目盛り
plt.rcParams['font.size'] = 16

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
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    def run_demographic(self, a):
        dfs = load_data(0, True, a.seed)
        df = dfs['all']
        print(df)

    class RocArgs(CommonArgs):
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

    class CenterCropArgs(CommonArgs):
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

    class RocFoldsMeanArgs(CommonArgs):
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

                fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
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

    def train_gbm(self, df, seed):
        data = []
        ii = []
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
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(False)
                ],
                categorical_feature=[],
            )

            pred_valid = model.predict(x_valid, num_iteration=model.best_iteration)

            fpr, tpr, thresholds = skmetrics.roc_curve(y_valid, pred_valid)
            acc = [skmetrics.accuracy_score(y_valid, pred_valid>t) for t in thresholds]
            f1 = 2 * (acc * tpr) / (acc + tpr)
            precision, recall, __ = skmetrics.precision_recall_curve(y_valid, pred_valid)
            data.append({
                'fpr': fpr,
                'tpr': tpr,
                'acc': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            })
            i = model.feature_importance(importance_type='gain')
            # i = pd.DataFrame(data=i, index=cols_measure)
            ii.append(i)

        data = pd.DataFrame(data)
        ii = np.array(ii)
        ii = pd.DataFrame(data=ii, columns=cols_measure, index=[f'fold{f}' for f in range(1,7)])
        return data, ii

    class GbmCurveByFoldsArgs(CommonArgs):
        curve: str = Field(..., regex=r'^roc|pr$')
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_gbm_curve_by_folds(self, a):
        # df = pd.read_excel('data/table.xlsx', index_col=0)
        dfs = load_data(0, True, a.seed)
        df = dfs['all']

        data, ii = self.train_gbm(df, a.seed)
        ii.to_excel('out/importance.xlsx')

        plt.figure(figsize=(6, 5), dpi=300)

        if a.curve == 'roc':
            xx, yy = data['fpr'], data['tpr'],
            draw = self.draw_roc_common
        else:
            xx, yy = data['precision'], data['recall'],
            draw = self.draw_pr_common
        self.draw_curve_with_ci(
            xx, yy,
            std_scale=1,
            color=['forestgreen', 'lightgreen'],
        )
        draw(
            title='Setting: B\nClinical measurements only',
            name=f'{a.curve}_b.png',
            show=not a.noshow
        )


    class ImageCurveByFoldsArgs(CommonArgs):
        curve: str = Field(..., regex=r'^roc|pr$')
        mode: str = 'image'
        depth: str = 'b0'
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_image_curve_by_folds(self, a):
        # dfs = load_data(0, True, a.seed)
        # df = dfs['all']

        base = 'data/result/classification_effnet_final'
        data = []
        for fold in range(1,7):
            pred_path = f'{base}/{a.mode}/tf_efficientnet_{a.depth}_fold{fold}/predictions.pt'
            pred = torch.load(pred_path, map_location=torch.device('cpu'))
            gts = pred['val_gts'].flatten().numpy()
            preds = pred['val_preds'].flatten().numpy()
            fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
            acc = [skmetrics.accuracy_score(gts, preds>t) for t in thresholds]
            f1 = 2 * (acc * tpr) / (acc + tpr)
            precision, recall, thresholds = skmetrics.precision_recall_curve(gts, preds)
            data.append({
                'fpr': fpr,
                'tpr': tpr,
                'acc': acc,
                'f1': f1,
                'recall': recall,
                'precision': precision,
            })
        data = pd.DataFrame(data)
        # print()
        # print(np.mean(data['auc']))

        if a.mode == 'image':
            name = f'{a.curve}_a_{a.depth}.png'
            title = f'Setting: A\nXp image only'
            color=['royalblue', 'lightblue']
        else:
            name = f'{a.curve}_c_{a.depth}.png'
            title = f'Setting: C\nXp image + Clinical measurements'
            color=['crimson', 'lightcoral']

        plt.figure(figsize=(6, 5), dpi=300)
        if a.curve == 'roc':
            xx, yy = data['fpr'], data['tpr']
            draw = self.draw_roc_common
        else:
            xx, yy = data['precision'], data['recall']
            draw = self.draw_pr_common

        self.draw_curve_with_ci(
            xx, yy,
            std_scale=1,
            color=color,
        )
        draw(
            title=title,
            name=name,
            show=not a.noshow
        )

    def load_ABC(self, depth, seed):
        base = 'data/result/classification_effnet_final'
        data_A = []
        data_C = []
        for fold in range(1,7):
            for mode, data in [('image', data_A), ('integrated', data_C)]:
                pred_path = f'{base}/{mode}/tf_efficientnet_{depth}_fold{fold}/predictions.pt'
                pred = torch.load(pred_path, map_location=torch.device('cpu'))
                gts = pred['val_gts'].flatten().numpy()
                preds = pred['val_preds'].flatten().numpy()
                fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
                acc = [skmetrics.accuracy_score(gts, preds>t) for t in thresholds]
                f1 = 2 * (acc * tpr) / (acc + tpr)
                precision, recall, __ = skmetrics.precision_recall_curve(gts, preds)
                data.append({
                    'fpr': fpr,
                    'tpr': tpr,
                    'acc': acc,
                    'f1': f1,
                    'thresholds': thresholds,
                    'recall': recall,
                    'precision': precision,
                })
        data_A = pd.DataFrame(data_A)
        data_C = pd.DataFrame(data_C)

        dfs = load_data(0, True, seed)
        df = dfs['all']
        data_B, __ii = self.train_gbm(df, seed)

        return data_A, data_B, data_C


    class CompareCurveArgs(CommonArgs):
        curve: str = Field(..., regex=r'^roc|pr$')
        depth: str = 'b0'
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_compare_curve(self, a):
        data_A, data_B, data_C = self.load_ABC(a.depth, a.seed)

        plt.figure(figsize=(6, 5), dpi=300)
        recipe = (
            ('A', data_A, ['royalblue', 'lightblue']),
            ('B', data_B, ['forestgreen', 'lightgreen']),
            ('C', data_C, ['crimson', 'lightcoral']),
        )
        for (code, data, c) in recipe:
            if a.curve == 'roc':
                xx, yy = data['fpr'], data['tpr']
            else:
                xx, yy = data['precision'], data['recall']
            self.draw_curve_with_ci(
                xx, yy, label='Setting '+code+' ({})',
                std_scale=1,
                color=c,
            )
        draw = self.draw_roc_common if a.curve == 'roc' else self.draw_pr_common
        draw(
            title=a.curve.upper() + ' curve',
            name=f'all_{a.curve}.png',
            show=not a.noshow,
        )

    class CompareAucArgs(CommonArgs):
        graph: str = Field(..., regex=r'^bar|box$')
        target: str = Field(..., regex=r'^roc|pr$')
        depth: str = 'b0'
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_compare_auc(self, a):
        dd = []
        for code, data in (('A', data_A), ('B', data_B), ('C', data_C)):
            if a.curve == 'roc':
                xx, yy = data['fpr'].values, data['tpr'].values
            else:
                xx, yy = data['recall'].values, data['precision'].values
            aucs = []
            for (x, y) in zip(xx, yy):
                aucs.append(skmetrics.auc(x, y))
            for i, auc in enumerate(aucs):
                dd.append({'auc': auc, 'setting': code, 'fold': i+1})
        data = pd.DataFrame(dd)
        # plt.figure(figsize=(6, 4), dpi=300)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        if a.graph == 'box':
            sns.boxplot(
                data=data, x='setting', y='auc', hue='setting',
                width=.5,
                # capsize=.1, errorbar=('ci', 95),
                # alpha=0.7,
                # linecolor=['royalblue', 'forestgreen', 'crimson'],
                palette=['lightblue', 'lightgreen', 'lightcoral'],
                boxprops=dict(alpha=.7),
                # errcolor='darkgrey',
                showfliers=False,
                ax=ax,
            )

            # sns.stripplot(
            sns.swarmplot(
                data=data, x='setting', y='auc', hue='setting',
                # palette=['lightblue', 'lightgreen', 'lightcoral'],
                alpha=0.7,
                palette=['grey'],
                # marker='X',
                size=5,
                ax=ax)
        elif a.graph == 'bar':
            sns.barplot(
                data=data, x='setting', y='auc', hue='setting',
                width=.5,
                # capsize=.1,
                errorbar=('ci', 95),
                err_kws={"color": "gray", "linewidth": 2.0, "alpha": 0.7},
                # edgecolor=['royalblue', 'forestgreen', 'crimson'],
                palette=['lightblue', 'lightgreen', 'lightcoral'],
                alpha=0.8,
                ax=ax,
            )

        else:
            raise RuntimeError(f'Invalid graph: {a.graph}')
        plt.ylabel(a.curve.upper() + ' AUC')
        # plt.grid(axis='y')
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.savefig(f'out/fig2/all_{a.curve}_{a.graph}.png')
        if not a.noshow:
            plt.show()

    class CompareMetricArgs(CommonArgs):
        metric: str = Field(..., regex=r'^acc|f1$')
        depth: str = 'b0'
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_compare_metric(self, a):
        data_A, data_B, data_C = self.load_ABC(a.depth, a.seed)
        for data, setting in [(data_A, 'A'), (data_B, 'B'), (data_C, 'C'), ]:
            print(setting)
            for fold, row in data.iterrows():
                tpr, fpr, acc, f1 = row['tpr'], row['fpr'], row['acc'], row['f1']
                i = np.argmax(tpr - fpr)
                print(fold, 'acc', acc[i], 'f1', f1[i])


        # self.data_A = data_A
        # self.data_B = data_B
        # self.data_C = data_C


    def draw_curve_with_ci(self, xx, yy, fill=True, label='{}', color=['blue', 'lightblue'], std_scale=2):
        l = []
        mean_x = np.linspace(0, 1, 100)
        aucs = []
        positive = yy[0][0] < yy[0][-1]
        for (x, y) in zip(xx, yy):
            l.append(np.interp(mean_x, x, y))
            if positive:
                aucs.append(skmetrics.auc(x, y))
            else:
                aucs.append(skmetrics.auc(y, x))

        yy = np.array(l)
        mean_y = yy.mean(axis=0)
        std_y = yy.std(axis=0)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        auc_label = f'AUC:{mean_auc:0.3f} ± {std_auc:0.3f}'
        plt.plot(mean_x, mean_y, label=label.format(auc_label), color=color[0])

        if fill:
            plt.fill_between(
                mean_x,
                mean_y - std_scale * std_y,
                mean_y + std_scale * std_y,
                color=color[1], alpha=0.2, label='± 1.0 s.d.')


    def draw_roc_common(self, title, name=None, show=False):
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('1 - Specificity')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.subplots_adjust(bottom=0.15, left=0.15, top=0.85)
        if name:
            plt.savefig(J('out/fig2', name))
        if show:
            plt.show()

    def draw_pr_common(self, title, name=None, show=False):
        plt.plot([0, 1], [1, 0], linestyle='--', color='gray', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.title(title)
        plt.subplots_adjust(bottom=0.15, left=0.15, top=0.85)
        if name:
            plt.savefig(J('out/fig2', name))
        if show:
            plt.show()

    class CsorArgs(CommonArgs):
        target: str = Field(..., regex=r'^mean|max$')


    def run_csor(self, a):
        df_org =  pd.read_excel('data/cams/powers_aggr.xlsx', usecols=list(range(1, 31)), index_col=0)
        col_value = f'{a.target} CSoR'

        def select_col(col, name):
            d = df_org[[col]] \
                .dropna() \
                .rename(columns={col: col_value})
            d['name'] = name
            return d

        suffix = '' if a.target == 'mean' else ' max'

        mean_recipe = [
            {
                'name': 'Bilateral',
                'arm1': ['neg bilateral', 'Negative'],
                'arm2': ['pos bilateral', 'Positive'],
                'paired': False
            }, {
                'name': 'Positive',
                'arm1': ['healthy', 'Healthy'],
                'arm2': ['affected', 'Affected'],
                'paired': True,
            }, {
                'name': 'Negative',
                'arm1': ['neg left', 'Left'],
                'arm2': ['neg right', 'Right'],
                'paired': True,
            },
        ]

        max_recipe = [
            {
                'name': 'Bilateral',
                'arm1': ['neg bilateral max', 'Negative'],
                'arm2': ['pos bilateral max', 'Positive'],
                'paired': False
            }, {
                'name': 'Positive',
                'arm1': ['healthy max', 'Healthy'],
                'arm2': ['affected max', 'Affected'],
                'paired': True,
            }, {
                'name': 'Negative',
                'arm1': ['neg left max', 'Left'],
                'arm2': ['neg right max', 'Right'],
                'paired': True,
            # }, {
            #     'name': 'Total vs Healthy',
            #     'arm1': ['total max', 'Total'],
            #     'arm2': ['healthy max', 'Healthy'],
            #     'paired': False,
            }
        ]

        recipe = mean_recipe if a.target == 'mean' else max_recipe

        def plot(ax, data):
            # ax.axhline(y=1.0, color='grey', linewidth=.5)
            ax.grid(axis='y')
            sns.barplot(data=data, x='name', y=col_value, hue='name', ax=ax,
                        # edgecolor=['tab:blue', 'tab:orange'],
                        palette=['tab:blue', 'tab:orange'],
                        alpha=0.7,
                        capsize=.4,
                        errorbar=('ci', 95),
                        err_kws={'linewidth': 1},
                        )

            # sns.boxplot(data=data, x='name', y=col_value, ax=ax)

        sns.set_palette(sns.color_palette())
        fig, axes = plt.subplots(1, len(recipe), sharey=True, figsize=(6, len(recipe)*3), dpi=300)
        fig.suptitle(f'Comparison of {a.target} CAM Score on ROI')

        for i, r in enumerate(recipe):
            print(r)
            df1 = select_col(r['arm1'][0], r['arm1'][1])
            df2 = select_col(r['arm2'][0], r['arm2'][1])
            df_c = pd.concat([df1, df2])
            print(r['name'], r['arm1'][1], 'vs', r['arm2'][1])
            print(df1[col_value])
            print(df2[col_value])
            if r['paired']:
                __, p  = stats.wilcoxon(df1[col_value], df2[col_value], alternative='two-sided')
                print('\t wilcoxon', p)
            else:
                __, p = stats.mannwhitneyu(df1[col_value], df2[col_value], alternative='two-sided')
                print('\t U test', p)
            ax = axes[i]
            plot(ax, df_c)
            ax.set(xlabel=r['name'])

        plt.savefig(f'out/fig3/bars_{a.target}.png')
        plt.show()

    def run_surface(self, a):
        id = 32
        side = 'left'
        roi = read_label_as_df(f'data/labels/{id:04}.txt')
        target_label = [1,2,3] if side == 'right' else [4,5,6]
        brois = roi[roi['label'].isin(target_label)]
        hroi = np.array([
            brois['x0'].min(),
            brois['y0'].min(),
            brois['x1'].max(),
            brois['y1'].max(),
        ]).round().astype(int)


        mask = Image.open(f'data/cams/crop/{id:04}_mask_pos.png')
        mask = np.array(mask) / 255

        mask = mask / mask.sum() * mask.shape[0] * mask.shape[1]

        fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot(121)
        im = ax.imshow(mask, cmap='jet')

        ax = fig.add_subplot(122, projection='3d')
        y = np.arange(mask.shape[0])
        x = np.arange(mask.shape[1])
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(Y, X, mask, cmap='jet', linewidth=0.01)
        # surf = ax.plot_trisurf(Y, X, mask, cmap='jet')

        # mean_surf = ax.plot_surface(Y, X, np.ones(mask.shape), cstride=2, rstride=1, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.5)

        # surf1 = ax.plot_surface(X, Y, mask,
        #                         cstride=2, rstride=1,
        #                         cmap=cm.Oranges, linewidth=0, antialiased=False, alpha=0.3)

        # surf2 = ax.plot_surface(X, Y, np.ones(mask.shape),
        #                         cstride=2, rstride=1,
        #                         cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.3)

        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # fig.colorbar(surf, ax=ax, aspect=5)
        plt.show()



    def run_s(self, a):
        mask = Image.open(f'data/cams/crop/0032_mask_pos.png')
        mask.resize((mask.width//3, mask.height//3))
        data = np.array(mask) / 255

        x, y = np.meshgrid(np.arange(data.shape[1] + 1), np.arange(data.shape[0] + 1))
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)

        # ポリゴンの頂点リストを作成
        vertices = list(zip(x, y, z))

        polygons = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # 各セルの4つの頂点のインデックスを計算
                v0 = i * (data.shape[1] + 1) + j
                v1 = v0 + 1
                v2 = v0 + data.shape[1] + 1
                v3 = v1 + data.shape[1] + 1

                # 頂点を指定してポリゴンを作成
                polygon = [vertices[v0], vertices[v1], vertices[v2], vertices[v3]]

                # カラーマップから色を取得し、ポリゴンの色を設定
                color = plt.cm.viridis(data[i, j])  # カラーマップを選択
                poly = Poly3DCollection([polygon], facecolors=[color], edgecolors='k')
                polygons.append(poly)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for poly in polygons:
            ax.add_collection3d(poly)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()





if __name__ == '__main__':
    cli = CLI()
    cli.run()

