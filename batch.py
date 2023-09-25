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
import matplotlib.cm as cm
from sklearn import metrics as skmetrics
from scipy import stats
import numpy as np
from pydantic import Field
import lightgbm as lgb

from endaaman.ml import BaseMLCLI, roc_auc_ci

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

plt.rcParams['xtick.direction'] = 'in' #x軸の目盛りの向き
plt.rcParams['ytick.direction'] = 'in' #y軸の目盛りの向き
plt.rcParams["xtick.minor.visible"] = True  #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True  #y軸補助目盛りの追加
plt.rcParams['xtick.bottom'] = True  #x軸の上部目盛り
plt.rcParams['ytick.left'] = True  #y軸の右部目盛り

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


    class GbmRocByFoldsArgs(CommonArgs):
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_gbm_roc_by_folds(self, a):
        # df = pd.read_excel('data/table.xlsx', index_col=0)
        dfs = load_data(0, True, a.seed)
        df = dfs['all']

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
            data.append({
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
            })

            i = model.feature_importance(importance_type='gain')
            # i = pd.DataFrame(data=i, index=cols_measure)
            ii.append(i)
            print(auc)

        print()
        data = pd.DataFrame(data)
        print(np.mean(data['auc']))

        ii = np.array(ii)
        ii = pd.DataFrame(data=ii, columns=cols_measure, index=[f'fold{f}' for f in range(1,7)])
        ii.to_excel('out/importance.xlsx')
        print(ii)

        self.draw_roc_with_ci(
            data['fpr'], data['tpr'],
            std_scale=1,
            title='Setting: B\nClinical measurements only',
            name='roc_b.png',
            color=['forestgreen', 'lightgreen'],
            show=not a.noshow
        )


    class ImageRocByFoldsArgs(CommonArgs):
        mode: str = 'image'
        depth: str = 'b0'
        noshow: bool = Field(False, cli=('--noshow', ))

    def run_image_roc_by_folds(self, a):
        # dfs = load_data(0, True, a.seed)
        # df = dfs['all']

        base = 'data/result/classification_effnet_final'
        data = []
        for fold in range(1,7):
            pred_path = f'{base}/{a.mode}/tf_efficientnet_{a.depth}_fold{fold}/predictions.pt'
            pred = torch.load(pred_path, map_location=torch.device('cpu'))
            gts = pred['val_gts'].flatten()
            preds = pred['val_preds'].flatten()
            fpr, tpr, thresholds = skmetrics.roc_curve(gts.numpy(), preds.numpy())
            auc = skmetrics.auc(fpr, tpr)
            data.append({
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
            })
            print(auc)
        data = pd.DataFrame(data)
        print()
        print(np.mean(data['auc']))

        if a.mode == 'image':
            name = f'roc_a_{a.depth}.png'
            title = f'Setting: A\nXp image only'
            color=['royalblue', 'lightblue']
        else:
            name = f'roc_c_{a.depth}.png'
            title = f'Setting: C\nXp image + Clinical measurements'
            color=['crimson', 'lightcoral']

        self.draw_roc_with_ci(
            data['fpr'], data['tpr'],
            std_scale=1,
            title=title,
            color=color,
            name=name,
            show=not a.noshow
        )


    def draw_roc_with_ci(self, fprs, tprs, title='{}',
                         color=['blue', 'lightblue'], std_scale=2, name=None, show=False):
        l = []
        mean_fpr = np.linspace(0, 1, 100)
        aucs = []
        for (fpr, tpr) in zip(fprs, tprs):
            l.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(skmetrics.auc(fpr, tpr))

        tprs = np.array(l)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)

        plt.figure(figsize=(6, 5), dpi=300)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, label=f'AUC:{mean_auc:0.3f} ± {std_auc:0.3f}', color=color[0])

        plt.fill_between(
            mean_fpr,
            mean_tpr - std_scale * std_tpr,
            mean_tpr + std_scale * std_tpr,
            color=color[1], alpha=0.5, label='± 1.0 s.d.')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Sensitivity')
        plt.xlabel('1 - Specificity')
        plt.legend(loc='lower right')
        plt.title(title.format('ROC curve'))
        if name:
            plt.savefig(J('out/fig2_rocs', name))
        if show:
            plt.show()


    def run_plot_powers(self, a):
        df_org = pd.read_excel('data/cams/powers_aggr.xlsx', usecols=list(range(7))+list(range(8,17)), index_col=0)
        print(df_org.columns)

        col_value = 'CAM score'

        # bilateral Positive vs Negative
        df_pos_bilateral = df_org[['pos bilateral']] \
            .dropna() \
            .rename(columns={'pos bilateral': col_value})
        df_pos_bilateral['name'] = 'Positive'

        df_neg_bilateral = df_org[['neg bilateral']] \
            .dropna() \
            .rename(columns={'neg bilateral': col_value})
        df_neg_bilateral['name'] = 'Negative'

        df_bilateral = pd.concat([df_neg_bilateral, df_pos_bilateral])
        u = stats.mannwhitneyu(df_pos_bilateral[col_value], df_neg_bilateral[col_value], alternative='two-sided')
        print('Bilateral / U test', u)

        # affected vs healthy
        df_affected = df_org[['affected']] \
            .dropna() \
            .rename(columns={'affected': col_value})
        df_affected['name'] = 'Affected'

        df_healthy = df_org[['healthy']] \
            .dropna() \
            .rename(columns={'healthy': col_value})
        df_healthy['name'] = 'Healthy'

        df_side = pd.concat([df_healthy, df_affected])
        u = stats.wilcoxon(df_affected[col_value], df_healthy[col_value], alternative='two-sided')
        print('Affected vs Healthy / wilcoxon', u)


        # negative: left vs right
        df_neg_left = df_org[['neg left']] \
            .dropna() \
            .rename(columns={'neg left': col_value})
        df_neg_left['name'] = 'Left'

        df_neg_right = df_org[['neg right']] \
            .dropna() \
            .rename(columns={'neg right': col_value})
        df_neg_right['name'] = 'Right'

        df_lr = pd.concat([df_neg_left, df_neg_right])
        u = stats.wilcoxon(df_neg_left[col_value], df_neg_right[col_value], alternative='two-sided')
        print('Negative Left vs Right / wilcoxon', u)


        def plot(ax, data):
            # ax.axhline(y=1.0, color='grey', linewidth=.5)
            # ax.grid(axis='y')
            sns.barplot(data=data, x='name', y=col_value, ax=ax,
                        capsize=.1, errorbar='ci',
                        # errcolor='darkgrey',
                        )

        sns.set_palette(sns.color_palette())
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6, 8), dpi=300)
        fig.suptitle('Comparison of CAM scores')

        ax = axes[0]
        plot(ax, df_bilateral)
        ax.set(xlabel='Bilateral')

        ax = axes[1]
        plot(ax, df_side)
        ax.set(xlabel='Positive cases', ylabel=None)

        ax = axes[2]
        plot(ax, df_lr)
        ax.set(xlabel='Negative cases', ylabel=None)

        plt.savefig('out/fig3_cam_score/bars.png')
        plt.show()



if __name__ == '__main__':
    cli = CLI()
    cli.run()

