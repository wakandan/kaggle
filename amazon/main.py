from torch import nn
from torch.utils.data import DataLoader, Dataset
from vanhelsing.logger import *
import pandas as pd
from pathlib import Path

ROOT = Path('/data/dataset/amazon')

df = pd.read_csv(ROOT / 'train_v2.csv')

print(df.head())


class AmazonDataset(Dataset):

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __add__(self, other):
        return super().__add__(other)
