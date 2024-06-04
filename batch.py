import os
from glob import glob
import math
import json
from itertools import combinations

import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt, animation, colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as mp3d
from vistats import annotate_brackets
from sklearn import metrics as skmetrics
from scipy import stats
import numpy as np
from pydantic import Field

import shap
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from endaaman import with_wrote
from endaaman.ml import BaseMLCLI, roc_auc_ci

from datasets import read_label_as_df
from common import col_target, cols_feature, cols_measure, load_data, col_to_label


# matplotlib.use('TkAgg')


sns.set_style('white')
sns.set_palette('tab10')
# default_palette = sns.color_palette()
# brightened_palette = sns.color_palette("pastel", n_colors=len(default_palette) - 2)
# sns.set_palette(brightened_palette)

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。

# plt.rcParams['xtick.direction'] = 'in' #x軸の目盛りの向き
# plt.rcParams["xtick.minor.visible"] = True  #x軸補助目盛りの追加
# plt.rcParams['xtick.bottom'] = True  #x軸の上部目盛り
plt.rcParams['ytick.direction'] = 'in' #y軸の目盛りの向き
plt.rcParams['ytick.minor.visible'] = True  #y軸補助目盛りの追加
plt.rcParams['ytick.left'] = True  #y軸の右部目盛り
plt.rcParams['font.size'] = 12
plt.rcParams['text.color'] = 'black'

json.encoder.FLOAT_REPR = lambda x: print(x) or format(x, '.8f')
J = os.path.join

def significant(v):
    if v < 0.001:
        return '***'
    if v < 0.01:
        return '**'
    if v < 0.05:
        return '*'
    return 'n.s.'

def plot_significant(ax, values_A, values_B, values_C, margin=0):
    # A vs C
    __, p_A_C = stats.ttest_rel(values_A, values_C)
    print('A vs C', p_A_C)
    sig_A_C = significant(p_A_C)

    # B vs C
    __, p_B_C = stats.ttest_rel(values_B, values_C)
    sig_B_C = significant(p_B_C)
    print('B vs C', p_B_C)

    asterisk_tuples = [
        (2, 1, sig_B_C), # B vs C
        (2, 0, sig_A_C), # A vs C
    ]

    annotate_brackets(
        asterisk_tuples,
        center=np.arange(3),
        height=[np.max([values_A, values_B, values_C])]*3,
        color='gray',
        margin=0.01,
        fs=14,
        ax=ax,
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax+margin)


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
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
        noshow: bool = False
        model:str = 'tf_efficientnet_b0'
        basedir: str = 'out/classification_effnet_final'
        plot: bool = False

    def run_roc_folds_mean(self, a):
        matplotlib.use('Agg')
        arm1 = 'image'
        arm2 = 'integrated'
        # arm2 = 'additional'
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

        with open(f'data/result/roc_folds_mean_{a.model}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        if a.plot:
            plt.ylabel('Sensitivity')
            plt.xlabel('1 - Specificity')
            plt.legend(loc='lower right')
            plt.savefig(f'data/result/roc_folds_mean_{a.model}.png')
            plt.show()


    def run_cnn_metrics(self, a):
        modes = ['image', 'integrated']
        depths = ['b0', 'b4', 'b8']
        R = {}

        with pd.ExcelWriter('data/result/metrics_ac.xlsx') as writer:
            for mode in modes:
                R[mode] = {}
                for d in depths:
                    M = []
                    for fold in [1, 2, 3, 4, 5, 6]:
                        # dir = f'out/classification_effnet_final/image/tf_efficientnet_b0_fold1/predictions.pt'
                        pred_file = f'out/classification_effnet_final/{mode}/tf_efficientnet_{d}_fold{fold}/predictions.pt'
                        P = torch.load(pred_file)
                        pred = P['val_preds'].cpu().flatten().numpy()
                        gt = P['val_gts'].cpu().flatten().numpy()

                        fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
                        precision, recall, _ = skmetrics.precision_recall_curve(gt, pred)
                        youden_index = np.argmax(tpr - fpr)
                        pred = pred > thresholds[youden_index]
                        M.append({
                            'roc_auc': skmetrics.auc(fpr, tpr),
                            'pr_auc': skmetrics.auc(recall, precision),
                            'acc': skmetrics.accuracy_score(gt, pred),
                            'prec': skmetrics.precision_score(gt, pred),
                            'f1': skmetrics.f1_score(gt, pred),
                        })

                    R[mode][d] = M
                    df = pd.DataFrame(M)
                    df.to_excel(writer, sheet_name=f'{mode}_{d}')

        # with open(with_wrote(f'data/result/all_metrics.json'), 'w', encoding='utf-8') as f:
        #     json.dump(R, f, indent=2)

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
        shap.initjs()

        data = []
        ii = []
        M = []
        s = pd.read_excel('data/tables/side.xlsx', index_col=0)[['side']]
        s.index = s.index.astype(str).str.zfill(4)
        df = df.join(s)

        left_valuess = {'values': [], 'shap': []}
        right_valuess = {'values': [], 'shap': []}
        true_valuess = {'values': [], 'shap': []}
        all_valuess = {'values': [], 'shap': []}
        for fold in [1,2,3,4,5,6]:
            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]

            x_train = df_train[cols_measure]
            y_train = df_train[col_target]
            x_valid = df_valid[cols_measure]
            y_valid = df_valid[col_target]
            side_train = df_train['side']
            side_valid = df_valid['side']

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
            gt, pred = y_valid, pred_valid

            fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
            precision, recall, _ = skmetrics.precision_recall_curve(gt, pred)
            youden_index = np.argmax(tpr - fpr)
            pred = pred > thresholds[youden_index]

            M.append({
                'roc_auc': skmetrics.auc(fpr, tpr),
                'pr_auc': skmetrics.auc(recall, precision),
                'acc': skmetrics.accuracy_score(gt, pred),
                'prec': skmetrics.precision_score(gt, pred),
                'f1': skmetrics.f1_score(gt, pred),
            })

            # acc = [skmetrics.accuracy_score(y_valid, pred_valid>t) for t in thresholds]
            # f1 = 2 * (acc * tpr) / (acc + tpr)
            # precision, recall, __ = skmetrics.precision_recall_curve(y_valid, pred_valid)
            # data.append({
            #     'fpr': fpr,
            #     'tpr': tpr,
            #     'acc': acc,
            #     'f1': f1,
            #     'precision': precision,
            #     'recall': recall,
            # })
            i = model.feature_importance(importance_type='gain')
            # i = pd.DataFrame(data=i, index=cols_measure)
            ii.append(i)

            explainer = shap.TreeExplainer(model=model)

            # x_valid_shap = x_valid.reset_index(drop=True)
            # shap_values = explainer.shap_values(X=x_valid_shap)
            x_valid = x_valid.rename(columns=col_to_label)

            right_values = x_valid[side_valid == 'right'].reset_index(drop=True)
            right_shap = explainer.shap_values(X=right_values)
            right_valuess['values'].append(right_values)
            right_valuess['shap'].append(right_shap)

            left_values = x_valid[side_valid == 'left'].reset_index(drop=True)
            left_shap = explainer.shap_values(X=left_values)
            left_valuess['values'].append(left_values)
            left_valuess['shap'].append(left_shap)

            all_values = x_valid.reset_index(drop=True)
            all_shap = explainer.shap_values(X=all_values)
            all_valuess['values'].append(all_values)
            all_valuess['shap'].append(all_shap)

            true_values = x_valid.reset_index(drop=True)
            true_shap = explainer.shap_values(X=true_values)
            true_valuess['values'].append(true_values)
            true_valuess['shap'].append(true_shap)

            # shap.summary_plot(right_shap, right_values)
            # plt.show()

        shap.summary_plot(
                np.concatenate(right_valuess['shap']),
                pd.concat(right_valuess['values']),
                show=False)
        plt.savefig('out/shap/right.png')
        plt.clf()

        shap.summary_plot(
                np.concatenate(left_valuess['shap']),
                pd.concat(left_valuess['values']),
                show=False)
        plt.savefig('out/shap/left.png')
        plt.clf()

        shap.summary_plot(
                np.concatenate(all_valuess['shap']),
                pd.concat(all_valuess['values']),
                show=False)
        plt.savefig('out/shap/all.png')
        plt.clf()

        shap.summary_plot(
                np.concatenate(true_valuess['shap']),
                pd.concat(true_valuess['values']),
                show=False)
        plt.savefig('out/shap/true.png')
        plt.clf()

        data = pd.DataFrame(data)
        M = pd.DataFrame(M)
        ii = pd.DataFrame(data=np.array(ii), columns=cols_measure, index=[f'fold{f}' for f in range(1,7)])
        return M, ii


    def train_svm(self, df, seed):
        M = []
        for fold in [1,2,3,4,5,6]:
            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]

            x_train = df_train[cols_measure]
            y_train = df_train[col_target]
            x_valid = df_valid[cols_measure]
            y_valid = df_valid[col_target]

            param_list = [0.001, 0.01, 0.1, 1, 10]
            best_score = 0
            best_parameters = {}
            best_model = None
            for gamma in param_list:
                for C in param_list:
                    # model = SVC(kernel=self.svm_kernel, gamma=gamma, C=C, random_state=None)
                    model = SVC(kernel='rbf', gamma=gamma, C=C, probability=True)
                    model.fit(x_train, y_train)
                    score = model.score(x_valid, y_valid)
                    if score > best_score:
                        best_score = score
                        best_parameters = {'gamma' : gamma, 'C' : C}
                        best_model = model
            model = best_model

            pred_valid = model.predict_proba(x_valid)[:, 1]
            gt, pred = y_valid, pred_valid

            fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
            precision, recall, _ = skmetrics.precision_recall_curve(gt, pred)
            youden_index = np.argmax(tpr - fpr)
            pred = pred > thresholds[youden_index]
            M.append({
                'roc_auc': skmetrics.auc(fpr, tpr),
                'pr_auc': skmetrics.auc(recall, precision),
                'acc': skmetrics.accuracy_score(gt, pred),
                'prec': skmetrics.precision_score(gt, pred),
                'f1': skmetrics.f1_score(gt, pred),
            })

        return pd.DataFrame(M)

    def train_rf(self, df, seed):
        M = []
        for fold in [1,2,3,4,5,6]:
            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]

            x_train = df_train[cols_measure]
            y_train = df_train[col_target]
            x_valid = df_valid[cols_measure]
            y_valid = df_valid[col_target]

            clf = RandomForestClassifier(random_state=0)
            clf = clf.fit(x_train, y_train)
            pred_valid = clf.predict_proba(x_valid)[:, 1]
            gt, pred = y_valid, pred_valid

            fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
            precision, recall, _ = skmetrics.precision_recall_curve(gt, pred)
            youden_index = np.argmax(tpr - fpr)
            pred = pred > thresholds[youden_index]
            M.append({
                'roc_auc': skmetrics.auc(fpr, tpr),
                'pr_auc': skmetrics.auc(recall, precision),
                'acc': skmetrics.accuracy_score(gt, pred),
                'prec': skmetrics.precision_score(gt, pred),
                'f1': skmetrics.f1_score(gt, pred),
            })
        return pd.DataFrame(M)

    def train_dt(self, df, seed):
        M = []
        for fold in [1,2,3,4,5,6]:
            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]

            x_train = df_train[cols_measure]
            y_train = df_train[col_target]
            x_valid = df_valid[cols_measure]
            y_valid = df_valid[col_target]

            clf = DecisionTreeClassifier()
            clf = clf.fit(x_train, y_train)
            pred_valid = clf.predict_proba(x_valid)[:, 1]
            gt, pred = y_valid, pred_valid

            fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
            precision, recall, _ = skmetrics.precision_recall_curve(gt, pred)
            youden_index = np.argmax(tpr - fpr)
            pred = pred > thresholds[youden_index]
            M.append({
                'roc_auc': skmetrics.auc(fpr, tpr),
                'pr_auc': skmetrics.auc(recall, precision),
                'acc': skmetrics.accuracy_score(gt, pred),
                'prec': skmetrics.precision_score(gt, pred),
                'f1': skmetrics.f1_score(gt, pred),
            })

        plot_tree(clf)
        plt.show()
        plt.clf()
        return pd.DataFrame(M)


    def train_linear(self, df, seed):
        M = []
        for fold in [1,2,3,4,5,6]:
            df_train = df[df['fold'] != fold]
            df_valid = df[df['fold'] == fold]

            x_train = df_train[cols_measure]
            y_train = df_train[col_target]
            x_valid = df_valid[cols_measure]
            y_valid = df_valid[col_target]

            model = LogisticRegression()
            model.fit(x_train, y_train)
            pred_valid = model.predict_proba(x_valid)[:, 1]
            gt, pred = y_valid, pred_valid

            fpr, tpr, thresholds = skmetrics.roc_curve(gt, pred)
            precision, recall, _ = skmetrics.precision_recall_curve(gt, pred)
            youden_index = np.argmax(tpr - fpr)
            pred = pred > thresholds[youden_index]
            M.append({
                'roc_auc': skmetrics.auc(fpr, tpr),
                'pr_auc': skmetrics.auc(recall, precision),
                'acc': skmetrics.accuracy_score(gt, pred),
                'prec': skmetrics.precision_score(gt, pred),
                'f1': skmetrics.f1_score(gt, pred),
            })

        return pd.DataFrame(M)


    def run_tabular_metrics(self, a):
        dfs = load_data(0, True, a.seed)
        df = dfs['all']

        M_gbm, importance = self.train_gbm(df, a.seed)
        M_svm = self.train_svm(df, a.seed)
        M_rf = self.train_rf(df, a.seed)
        M_linear = self.train_linear(df, a.seed)
        # M_dt = self.train_dt(df, a.seed)

        with pd.ExcelWriter(with_wrote('data/result/metrics_b.xlsx')) as writer:
            M_gbm.to_excel(writer, sheet_name='LightGBM')
            M_svm.to_excel(writer, sheet_name='SVM')
            M_rf.to_excel(writer, sheet_name='RandomForest')
            M_linear.to_excel(writer, sheet_name='Linear')


    class GbmCurveByFoldsArgs(CommonArgs):
        curve: str = Field(..., regex=r'^roc|pr$')
        depth: str = Field('', regex=r'^b0|b4|b8$')
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
        draw(title='Setting: B\nClinical measurements only')

        os.makedirs(f'out/fig2/{a.depth}', exist_ok=True)
        plt.savefig(f'out/fig2/{a.depth}/{a.curve}_b.png')
        if not a.noshow:
            plt.show()

    def run_compare_importance(self, a):
        df = pd.read_excel('out/importance.xlsx', index_col=0)
        print(df)
        r = [
            ['a', 'Yamamuro A'],
            ['b', 'Yamamuro B'],
            ['alpha', 'Acetabular index'],
            ['oe', 'O-edge angle'],
        ]
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 5), dpi=300, sharey=True)

        for i, (name, label) in enumerate(r):
            ax = axes[i]
            left_values = df['left_' + name].values
            right_values = df['right_' + name].values
            data_left = pd.DataFrame({
                label: 'Left',
                'importance': left_values,
            })
            data_right = pd.DataFrame({
                label: 'Right',
                'importance': right_values,
            })
            sns.barplot(
                data=pd.concat([data_right, data_left]), x=label, y='importance', hue=label,
                width=.5,
                capsize=.2,
                errorbar=('ci', 95),
                err_kws={'color': 'gray', 'linewidth': 1.0},
                palette=['bisque', 'darkorange', ],
                alpha=0.8,
                ax=ax,
            )
            ax.set_ylabel('Importance')

            __, p_value = stats.ttest_rel(left_values, right_values)

            annotate_brackets(
                [
                    (0, 1, significant(p_value)),
                ],
                center=np.arange(2),
                height=[np.max([left_values, right_values])]*3,
                color='gray',
                margin=0.01,
                fs=14,
                ax=ax,
            )

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax+50)
        fig.tight_layout(pad=1.0)
        plt.subplots_adjust(bottom=0.15, left=0.2)
        os.makedirs('out/fig4', exist_ok=True)
        plt.savefig('out/fig4/importance.png')
        plt.show()


    class ImageCurveByFoldsArgs(CommonArgs):
        curve: str = Field(..., regex=r'^roc|pr$')
        mode: str = 'image'
        depth: str = Field(..., regex=r'^b0|b4|b8$')
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
            filename = f'{a.curve}_a.png'
            title = f'Setting: A\nXp image only'
            color=['royalblue', 'lightblue']
        else:
            filename = f'{a.curve}_c.png'
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
        draw(title=title)

        plt.savefig(J(f'out/fig2/{a.depth}', filename))
        if not a.noshow:
            plt.show()

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
        depth: str = Field(..., regex=r'^b0|b4|b8$')
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
        curve: str = Field(..., regex=r'^roc|pr$')
        depth: str = Field(..., regex=r'^b0|b4|b8$')
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_compare_auc(self, a:CompareAucArgs):
        data_A, data_B, data_C = self.load_ABC(a.depth, a.seed)
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
                dd.append({'value': auc, 'setting': code, 'fold': i+1})
        data = pd.DataFrame(dd)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

        if a.graph == 'box':
            sns.boxplot(
                data=data, x='setting', y='value', hue='setting',
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

            sns.swarmplot(
            # sns.stripplot(
                data=data, x='setting', y='value', hue='setting',
                # palette=['lightblue', 'lightgreen', 'lightcoral'],
                alpha=0.7,
                palette=['grey']*3,
                # marker='X',
                size=5,
                ax=ax,
            )

        elif a.graph == 'bar':
            bars = sns.barplot(
                data=data, x='setting', y='value', hue='setting',
                width=.5,
                capsize=.2,
                errorbar=('ci', 95),
                err_kws={"color": "gray", "linewidth": 1.0},
                # edgecolor=['royalblue', 'forestgreen', 'crimson'],
                palette=['lightblue', 'lightgreen', 'lightcoral'],
                alpha=0.8,
                ax=ax,
            )
        else:
            raise RuntimeError(f'Invalid graph: {a.graph}')


        values_A = data[data['setting'] == 'A']['value']
        values_B = data[data['setting'] == 'B']['value']
        values_C = data[data['setting'] == 'C']['value']
        plot_significant(ax, values_A, values_B, values_C, (0.05 if a.graph == 'bar' else 0.015))

        if a.curve == 'pr':
            title = 'AUPRC'
        else:
            title = 'AUROC'
        plt.ylabel(title)
        plt.title(title)
        # plt.grid(axis='y')
        # ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.15, left=0.2)
        plt.savefig(f'out/fig2/{a.depth}/all_{a.graph}_{a.curve}.png')
        if not a.noshow:
            plt.show()

    class CompareMetricArgs(CommonArgs):
        metric: str = Field(..., regex=r'^acc|f1$')
        graph: str = Field(..., regex=r'^bar|box$')
        depth: str = Field(..., regex=r'^b0|b4|b8$')
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_compare_metric(self, a):
        data_A, data_B, data_C = self.load_ABC(a.depth, a.seed)
        dd = []
        for data, setting in [(data_A, 'A'), (data_B, 'B'), (data_C, 'C'), ]:
            for fold, row in data.iterrows():
                tpr, fpr, acc, f1 = row['tpr'], row['fpr'], row['acc'], row['f1']
                # i = np.argmax(f1)
                i = np.argmax(tpr - fpr)
                value = acc if a.metric == 'acc' else f1
                dd.append({'value': np.max(value), 'setting': setting, 'threshold': i+1})

        data = pd.DataFrame(dd)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        if a.graph == 'box':
            sns.boxplot(
                data=data, x='setting', y='value', hue='setting',
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

            sns.swarmplot(
            # sns.stripplot(
                data=data, x='setting', y='value', hue='setting',
                # palette=['lightblue', 'lightgreen', 'lightcoral'],
                alpha=0.7,
                palette=['gray']*3,
                # marker='X',
                size=5,
                ax=ax,
            )
        elif a.graph == 'bar':
            bars = sns.barplot(
                data=data, x='setting', y='value', hue='setting',
                width=.5,
                capsize=.2,
                errorbar=('ci', 95),
                err_kws={"color": "gray", "linewidth": 1.0},
                # edgecolor=['royalblue', 'forestgreen', 'crimson'],
                palette=['lightblue', 'lightgreen', 'lightcoral'],
                alpha=0.8,
                ax=ax,
            )
        else:
            raise RuntimeError(f'Invalid graph: {a.graph}')

        values_A = data[data['setting'] == 'A']['value']
        values_B = data[data['setting'] == 'B']['value']
        values_C = data[data['setting'] == 'C']['value']

        plot_significant(ax, values_A, values_B, values_C, (0.05 if a.graph == 'bar' else 0.015))

        # plt.grid(axis='y')
        title = {'acc': 'Accuracy', 'f1': 'F1 score'}[a.metric]
        plt.title(title)
        plt.ylabel(title)
        # ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.15, left=0.2)
        os.makedirs(f'out/fig2/{a.depth}', exist_ok=True)
        plt.savefig(f'out/fig2/{a.depth}/all_{a.graph}_{a.metric}.png')
        data.to_excel(f'out/fig2/{a.depth}/{a.metric}.xlsx')
        if not a.noshow:
            plt.show()


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


    def draw_roc_common(self, title):
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('1 - Specificity')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.subplots_adjust(bottom=0.15, left=0.15, top=0.85)

    def draw_pr_common(self, title):
        plt.plot([0, 1], [1, 0], linestyle='--', color='gray', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.title(title)
        plt.subplots_adjust(bottom=0.15, left=0.15, top=0.85)

    class CsorArgs(CommonArgs):
        target: str = Field(..., regex=r'^mean|max$')
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_csor(self, a:CsorArgs):
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
                'name': 'All cases',
                'arm1': ['neg bilateral', 'Negative'],
                'arm2': ['pos bilateral', 'Positive'],
                'palette': ['bisque', 'darkorange'],
                'paired': False
            }, {
                'name': 'Positive cases',
                'arm1': ['healthy', 'Contra'],
                'arm2': ['affected', 'Affected'],
                'palette': ['bisque', 'darkorange'],
                'paired': True,
            }, {
                'name': 'Negative cases',
                'arm1': ['neg right', 'Right'],
                'arm2': ['neg left', 'Left'],
                'palette': ['bisque', 'bisque'],
                'paired': True,
            },
        ]

        max_recipe = [
            {
                'name': 'All cases',
                'arm1': ['neg bilateral max', 'Negative'],
                'arm2': ['pos bilateral max', 'Positive'],
                'palette': ['bisque', 'darkorange'],
                'paired': False
            }, {
                'name': 'Positive cases',
                'arm1': ['healthy max', 'Contra'],
                'arm2': ['affected max', 'Affected'],
                'palette': ['bisque', 'darkorange'],
                'paired': True,
            }, {
                'name': 'Negative cases',
                'arm1': ['neg right max', 'Right'],
                'arm2': ['neg left max', 'Left'],
                'palette': ['bisque', 'bisque'],
                'paired': True,
            }
        ]

        recipe = mean_recipe if a.target == 'mean' else max_recipe

        sns.set_palette(sns.color_palette())
        fig, axes = plt.subplots(1, len(recipe), sharey=True, figsize=(len(recipe)*3, 6), dpi=300)
        fig.suptitle(f'CSoR {a.target}')

        for i, r in enumerate(recipe):
            df1 = select_col(r['arm1'][0], r['arm1'][1])
            df2 = select_col(r['arm2'][0], r['arm2'][1])
            df_c = pd.concat([df1, df2])
            print(r['name'], r['arm1'][1], 'vs', r['arm2'][1])
            if r['paired']:
                __, p  = stats.wilcoxon(df1[col_value], df2[col_value], alternative='two-sided')
                print('\t wilcoxon', p)
            else:
                __, p = stats.mannwhitneyu(df1[col_value], df2[col_value], alternative='two-sided')
                print('\t U test', p)
            ax = axes[i]
            sns.barplot(
                data=df_c, x='name', y=col_value, hue='name',
                width=.5,
                capsize=.2,
                errorbar=('ci', 95),
                err_kws={'color': 'gray', 'linewidth': 1.0},
                palette=r['palette'],
                alpha=0.8,
                ax=ax,
            )
            ax.set(xlabel=r['name'], ylabel=f'CSoR {a.target}')

            asterisk_tuples = [
                (0, 1, significant(p)),
            ]

            margin = 0.2 if a.target == 'mean' else 0.4

            annotate_brackets(
                asterisk_tuples,
                center=np.arange(2),
                height=[np.max([ df1[col_value].mean(), df2[col_value].mean()] ) + margin*2] * 2,
                color='gray',
                margin=-0.1,
                fs=14,
                ax=ax,
            )
        # use last ax
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax+margin)

        plt.subplots_adjust(bottom=0.15, left=0.2)
        os.makedirs('out/fig3', exist_ok=True)
        plt.savefig(f'out/fig3/bars_{a.target}.png')
        if not a.noshow:
            plt.show()

    class SurfaceArgs(CommonArgs):
        id: int = 44
        animate: bool = Field(False, cli=('--animate', ))
        cmap: str = 'jet'
        grayed: bool = Field(False, cli=('--grayed', ))

    def run_surface(self, a):
        roi = read_label_as_df(f'data/labels/{a.id:04}.txt')
        # target_label = [1,2,3] if side == 'right' else [4,5,6]
        hrois = {}
        for side, idx in (('left', [4,5,6]), ('right', [1,2,3])):
            brois = roi[roi['label'].isin(idx)]
            hroi = np.array([
                brois['x0'].min(),
                brois['y0'].min(),
                brois['x1'].max(),
                brois['y1'].max(),
            ]).round().astype(int)
            # offset for center 512px
            hroi -= (624 - 512)//2
            hrois[side] = hroi

        left_hroi = hrois['left']
        right_hroi = hrois['right']

        mask = Image.open(f'data/cams/crop/{a.id:04}_mask_pos.png')
        mask = np.array(mask) / 255

        mask = mask / mask.sum() * mask.shape[0] * mask.shape[1]

        fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot(121)
        im = ax.imshow(mask, cmap=a.cmap)

        ax = fig.add_subplot(122, projection='3d')
        y = np.arange(mask.shape[0])
        x = np.arange(mask.shape[1])
        X, Y = np.meshgrid(x, y)

        cmap = plt.get_cmap(a.cmap)
        normalize = mcolors.Normalize(vmin=np.min(mask), vmax=np.max(mask))

        # REGIONAL
        if a.grayed:
            colors = np.full((X.shape[0], X.shape[0], 4), (.0, .0, .0, .0), dtype=float)
            for yi, yv in enumerate(y):
                for xi, xv in enumerate(x):
                    v = mask[yi, xi]
                    c = np.array(cmap(normalize(v)))
                    colors[yi, xi] = c
                    if left_hroi[0] < xi < left_hroi[2] and left_hroi[1] < yi < left_hroi[3]:
                        continue
                    if right_hroi[0] < xi < right_hroi[2] and right_hroi[1] < yi < right_hroi[3]:
                        continue
                    # c = np.clip(c - np.array([.5]*3 + [.0]), 0.0, 1.0)
                    # c[-1] = 0.8
                    colors[yi, xi] = np.array([.4, .4, .4, 0.1])
                    # colors[yi, xi] = c

            surf = ax.plot_surface(Y, X, mask,
                                   linewidth=0.5,
                                   facecolors=colors,
                                   shade=False,
                                   cmap='jet')
        else:
            surf = ax.plot_surface(Y, X, mask, linewidth=0.01, cmap='jet', antialiased=False, edgecolors=np.array([.1, .1, .1, 0.1]))
            # surf = ax.plot_wireframe(Y, X, mask, linewidth=0.5, cmap='jet', antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.view_init(elev=30, azim=30)

        if not a.animate:
            plt.savefig('out/fig3/surface.png')
            plt.show()
            return

        def angle(i):
            # azimuth angle : 0 deg to 360 deg
            ax.view_init(elev=20, azim=i*4)
            return fig,

        ani = animation.FuncAnimation(fig, angle, init_func=None, frames=90, interval=50, blit=True)
        ani.save('out/fig3/surface.gif', writer='imagemagick', fps=1000/50)
        ani.save('out/fig3/surface.mp4', writer='ffmpeg', fps=1000/50)



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

