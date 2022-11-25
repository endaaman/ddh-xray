import numpy as np

col_target = 'treatment'
cols_measure = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

# cols_cat = ['female', 'family_history', 'breech_presentation', 'skin_laterality', 'limb_limitation']
# cols_val = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

cols_cat = []
cols_val = ['female', 'breech_presentation'] + cols_measure
# cols_val = ['female', 'breech_presentation', 'family_history'] + cols_measure

do_abs = lambda x: np.power(x, 2)
# do_abs = lambda x: x
cols_extend = {
    # 'alpha_diff': lambda x: do_abs(x['left_alpha'] - x['right_alpha']),
    # 'oe_diff': lambda x: do_abs(x['left_oe'] - x['right_oe']),
    # 'a_diff': lambda x: do_abs(x['left_a'] - x['right_a']),
    # 'b_diff': lambda x: do_abs(x['left_b'] - x['right_b']),
}
cols_feature = cols_cat + cols_val + list(cols_extend.keys())

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

def extend_table(df):
    for col, fn in cols_extend.items():
        df[col] = fn(df)
