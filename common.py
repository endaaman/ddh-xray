import pandas as pd

df = pd.read_excel('data/table.xlsx')
df_train = df[df['test'] == 0]
df_test = df[df['test'] == 1]

col_target = 'treatment'
cols_feature = ['sex', 'family_history', 'breech_presentation', 'skin_laterality', ' limb_limitation']
cols_cat = cols_feature
