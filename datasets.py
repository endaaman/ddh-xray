# from tqdm import tqdm
# import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# import torch
from torch.utils.data import Dataset, DataLoader


class PatientDataset(Dataset):
    def __init__(self, is_test=False):
        self.mm_df = pd.read_excel('data/measurement.xlsx')
        self.table_df = pd.read_excel('data/table.xlsx')
        # print(self.df[self.table_df['name']=='0147.jpg'])
        print(self.table_df)
        if


if __name__ == '__main__':
    ds = PatientDataset()
