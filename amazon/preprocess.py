import pandas as pd
from pathlib import Path
import h5py
import numpy as np
import random
from torchvision import transforms
from PIL import Image
import glob

ROOT = Path('/data/dataset/amazon')

CSV = ROOT / 'train_v2.csv'
OUT = ROOT / 'train.h5py'
TEST = ROOT / 'test'
SHUFFLE_DATA = True
RANDOM_SEED = 42
SPLIT_RATIO = .2
IMG_SIZE = 224

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

df = pd.read_csv(CSV)
df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg')
fns = list(df['image_name'])
fn_ids = np.array(range(len(df)))
id2fn = {i: fn for i, fn in enumerate(fns)}
fn2id = {fn: i for i, fn in enumerate(fns)}
print(f'first 10 file names = {fns[:10]}')
df['tag_list'] = df['tags'].apply(lambda x: x.split())
tags = sorted(list(set([i for sublist in list(df['tag_list']) for i in sublist])))
print(f'all tags = {tags}')
labels = np.zeros((len(df), len(tags)))
print(f'shape of labels = {labels.shape}')
if SHUFFLE_DATA:
    random.shuffle(fn_ids)
ids = fn_ids

test_fns = glob.glob(str(TEST / '*.jpg'))
test_fns = sorted(test_fns)
print(f'len of test fns = {len(test_fns)}')


def encode_tags(tag_list):
    res = np.zeros(len(tags))
    for i, tag in enumerate(tags):
        if tag in tag_list:
            res[i] = 1
    return res


for row in df.iterrows():
    i, row = row
    tag_list = row['tag_list']
    labels[i] = encode_tags(tag_list)

split = int(np.floor(SPLIT_RATIO * len(df)))
train_fns = [fns[i] for i in fn_ids[split:]]
val_fns = [fns[i] for i in fn_ids[:split]]
train_lbl = labels[fn_ids[split:]].astype(np.uint8)
val_lbl = labels[fn_ids[:split]].astype(np.uint8)
print(f'train lbl shape = {train_lbl.shape}')
print(f'val lbl shape = {val_lbl.shape}')

## Test the labels and splitting to make sure that the shuffle is working correctly
first_10_filenames = train_fns[:10]
first_10_ids = [fn2id[fn] for fn in first_10_filenames]
first_10_df = df.iloc[first_10_ids].copy(deep=False).reset_index()
first_10_df['encoded'] = first_10_df['tag_list'].apply(encode_tags)
for i, row in first_10_df.iterrows():
    assert np.array_equal(row['encoded'], train_lbl[i])

train_shape = (len(train_fns), 3, IMG_SIZE, IMG_SIZE)
val_shape = (len(val_fns), 3, IMG_SIZE, IMG_SIZE)
test_shape = (len(test_fns), 3, IMG_SIZE, IMG_SIZE)

trfs = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

with h5py.File(OUT, mode='w') as file:
    file.create_dataset('train_img', train_shape, compression='gzip', dtype=np.float)
    file.create_dataset('val_img', val_shape, compression='gzip', dtype=np.float)
    file.create_dataset('train_lbl', train_lbl.shape, compression='gzip', dtype=np.uint8)
    file.create_dataset('val_lbl', val_lbl.shape, compression='gzip', dtype=np.uint8)
    file.create_dataset('test_img', test_shape, compression='gzip', dtype=np.float)
    file['train_lbl'][...] = train_lbl
    file['val_lbl'][...] = val_lbl
    for i, fn in enumerate(train_fns):
        im = Image.open(ROOT / 'train' / fn)
        im = trfs(im)
        im = im[:3, :, :]
        file['train_img'][i, ...] = im
    for i, fn in enumerate(val_fns):
        im = Image.open(ROOT / 'train' / fn)
        im = trfs(im)
        im = im[:3, :, :]
        file['val_img'][i, ...] = im
    for i, fn in enumerate(test_fns):
        im = Image.open(fn)
        im = trfs(im)
        im = im[:3, :, :]
        file['test_img'][i, ...] = im

print('done')
