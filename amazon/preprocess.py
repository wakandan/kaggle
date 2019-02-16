import pandas as pd
from pathlib import Path
import h5py

ROOT = Path('/data/dataset/amazon')

CSV = ROOT / 'train_v2.csv'
OUT = ROOT / 'train.h5py'
df = pd.read_csv(CSV)
df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg')
df['tags'] = df['tags'].apply(lambda x: x.split())
tags = list(set([i for sublist in list(df['tags']) for i in sublist]))
print(tags)
print(df.head())
