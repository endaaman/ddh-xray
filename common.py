import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



col_target = 'treatment'
# cols_clinical = ['female', 'breech_presentation']
# cols_clinical = ['female', 'breech_presentation', 'family_history', 'skin_laterality', 'limb_limitation']
cols_clinical = ['female', 'breech_presentation', 'family_history']

cols_measure = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

do_abs = lambda x: np.power(x, 2)
cols_extend = {
    # 'alpha_diff': lambda x: do_abs(x['left_alpha'] - x['right_alpha']),
    # 'oe_diff': lambda x: do_abs(x['left_oe'] - x['right_oe']),
    # 'a_diff': lambda x: do_abs(x['left_a'] - x['right_a']),
    # 'b_diff': lambda x: do_abs(x['left_b'] - x['right_b']),
}

cols_feature = cols_measure + list(cols_extend.keys()) + cols_clinical

col_to_label = {
    'female': 'Female',
    'breech_presentation': 'Breech presentation',
    'left_a': 'Left A',
    'right_a': 'Right A',
    'left_b': 'Left B',
    'right_b': 'Right B',
    'left_alpha': 'Left α',
    'right_alpha': 'Right α',
    'left_oe': 'Left OE',
    'right_oe': 'Right OE',
}

def load_data(test_ratio, normalize_features, seed):
    df_all = pd.read_excel('data/tables/main.xlsx', index_col=0, converters={'label': str})
    df_ind = pd.read_excel('data/tables/ind.xlsx', index_col=0, converters={'label': str})

    # sheet = 'data'
    # sheet = 'filled'
    # df_measure = pd.read_excel('data/measurement_all.xlsx', converters={'label': str}, usecols=range(9), sheet_name=sheet)
    # df_measure['label'] = df_measure['label'].map('{:0>4}'.format)
    # df_measure = df_measure.set_index('label').fillna(0)
    # df_all = pd.merge(df_all, df_measure, left_index=True, right_index=True)

    ms = {}
    if normalize_features:
        for col in cols_measure:
            t = df_all[col]
            ms[col] = (t.mean(), t.std())
            # df_all[col] = (t - t.mean()) / t.std()
        for df in [df_all, df_ind]:
            for col in cols_measure:
                t = df[col]
                df[col] = (t - ms[col][0]) / ms[col][1]

    if test_ratio > 0:
        df_train, df_test = train_test_split(
            df_all,
            test_size=test_ratio,
            random_state=seed,
            stratify=df_all['treatment'],
        )
        df_test['test'] = 1
        df_train['test'] = 0
    elif test_ratio == 0:
        df_train = pd.DataFrame(columns=df_all.columns)
        df_test = pd.DataFrame(columns=df_all.columns)
    else:
        df_test = df_all[df_all['test'] > 0]
        df_train = df_all[df_all['test'] < 1]

    return {
        'all': df_all,
        'train': df_train,
        'test': df_test,
        'ind': df_ind,
    }
