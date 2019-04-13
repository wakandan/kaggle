import glob
import sys

from torch.autograd import Variable

sys.path.append('..')

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from vanhelsing.logger import *
import pandas as pd
from pathlib import Path
import h5py
from torchvision import models, transforms
import numpy as np
import random
from PIL import Image
from vanhelsing.utils_train import *
from vanhelsing.configurator import *
from torch import functional as F

parser = TrainArgumentParser()

parser.add_argument('--train', action='store_true')
parser.add_argument('--model')
parser.add_argument('--image')
parser.add_argument('--freeze', type=int, default=0)
parser.add_argument('--column')

args = parser.parse_args()

ROOT = Path('/home/kdang/dataset/amazon')

CSV = ROOT / 'train.csv'
OUT = ROOT / 'train.h5py'
TEST = ROOT / 'test'
SHUFFLE_DATA = True
RANDOM_SEED = args.random_seed
SPLIT_RATIO = args.split_ratio
IMG_SIZE = 224
TRAIN = args.train
IMAGE = args.image
FREEZE = args.freeze
RESUME = args.resume
MODEL = args.model
COLUMN = args.column

if COLUMN is not None:
    logging.info(f'training for column = {COLUMN}')

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

logging.info(f'CSV location: {CSV}')
df = pd.read_csv(CSV)
df['tag_list'] = df['tags'].apply(lambda x: x.split())
df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg')
fns = list(df['image_name'])
tags = sorted(list(set([i for sublist in list(df['tag_list']) for i in sublist])))

for tag in tags:
    df[tag] = pd.Series(np.zeros(len(df)))


def apply_tag(row):
    for tag in row['tag_list']:
        row[tag] = 1
    return row


df = df.apply(apply_tag, axis=1)

labels = np.zeros((len(df), len(tags)))
fn_ids = np.array(range(len(df)))
id2fn = {i: fn for i, fn in enumerate(fns)}
fn2id = {fn: i for i, fn in enumerate(fns)}

split = int(np.floor(SPLIT_RATIO * len(df)))
train_fns = [fns[i] for i in fn_ids[split:]]
val_fns = [fns[i] for i in fn_ids[:split]]
train_lbl = labels[fn_ids[split:]].astype(np.uint8)
val_lbl = labels[fn_ids[:split]].astype(np.uint8)

BS = args.bs
EPOCHS = args.epochs
LR = args.lr

trfs = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


def encode_tags(tag_list):
    res = np.zeros(len(tags))
    for i, tag in enumerate(tags):
        if tag in tag_list:
            res[i] = 1
    return res


def decode_tags(tags_prediction):
    result = []
    for i, j in enumerate(tags_prediction):
        if j == 1:
            result.append(tags[i])
    return result


for row in df.iterrows():
    i, row = row
    tag_list = row['tag_list']
    labels[i] = encode_tags(tag_list)


class AmazonDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels, column=None) -> None:
        super().__init__()
        self.df = df
        logging.info(f'original sample size = {len(self.df)}')
        self.column = column
        self.labels = labels
        if column is not None:
            self.df = self.df[['image_name', column]]
            logging.info(f"selecting on just 1 column {self.column}")
            logging.info(f'neg sample size = {len(self.df.loc[self.df[column] == 0])}')
            logging.info(f'pos sample size = {len(self.df.loc[self.df[column] == 1])}')
            ratio = len(self.df.loc[self.df[column] == 1]) / len(self.df.loc[self.df[column] == 0])
            if ratio < 1:  # less positive samples than negative samples
                less_samples = self.df.loc[self.df[column] == 1]
                more_samples = self.df.loc[self.df[column] == 0]
                logging.info(f'ratio pos/neg = {len(less_samples) / len(self.df)}')
            else:  # more positive samples than negative samples
                less_samples = self.df.loc[self.df[column] == 0]
                more_samples = self.df.loc[self.df[column] == 1]
                logging.info(f'ratio neg/pos = {len(less_samples) / len(self.df)}')
            logging.info(f'size of less sample potion: {len(less_samples)}')
            logging.info(f'size of more sample potion: {len(self.df) - len(less_samples)}')
            dup = int(np.floor((len(self.df) - len(less_samples)) // len(less_samples)) + 1)
            logging.info(f'duplication = {dup}')
            # only get just enough samples so that the ratio between new pos/neg is 50:50
            new_samples = pd.DataFrame(pd.np.tile(less_samples, (dup, 1)), columns=['image_name', column])[
                          :len(more_samples) - len(less_samples)]
            self.df = pd.concat([self.df, new_samples], axis=0)
            logging.info(f'new neg sample size = {len(self.df.loc[self.df[column] == 0])}')
            logging.info(f'new pos sample size = {len(self.df.loc[self.df[column] == 1])}')
            # logging.info(f'new neg sample size = {len(self.df.loc[self.df[column] == 0])}')
            # logging.info(f'new pos sample size = {len(self.df.loc[self.df[column] == 1])}')
            logging.info(f'target total sample size = {len(self.df)}')
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.labels = self.df[column]
            # logging.info("train data frame")
            # print(self.df.head())
            # logging.info('label data frame')
            # print(self.labels.head())

    def __getitem__(self, index):
        row = self.df.iloc[index]
        im_fn = row['image_name']
        im = Image.open(ROOT / f'train-jpg/{im_fn}')
        im = trfs(im)
        im = im[:3, :, :]
        lbl = torch.from_numpy(np.array(self.labels[index]))
        # lbl = np.zeros(len(tags) * 2)
        # for i, j in enumerate(labels[index]):
        #     if j == 0:
        #         lbl[i * 2] = 1
        #     else:
        #         lbl[i * 2 + 1] = 1
        # lbl = torch.from_numpy(lbl)
        return im, lbl

    def __len__(self):
        return len(self.df)


class AmazonTestDataset(Dataset):

    def __init__(self) -> None:
        self.files = glob.glob(str(ROOT / 'test-jpg/*.jpg'))
        self.files += glob.glob(str(ROOT / 'test-jpg-additional/*.jpg'))
        self.files = sorted(self.files)

    def __getitem__(self, index):
        im_fn = self.files[index]
        im = Image.open(im_fn)
        im = trfs(im)
        im = im[:3, :, :]
        return im_fn, im

    def __len__(self):
        return len(self.files)


class Net(nn.Module):

    def __init__(self, column=None):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.column = column
        if self.column is None:
            self.net.fc = nn.Sequential(
                # nn.Linear(512, 256, bias=True),
                nn.Linear(512, len(tags), bias=True),
                # nn.Linear(512, 256, bias=True),
                # nn.Linear(256, 1, bias=True),
            )
        else:
            self.net.fc = nn.Sequential(
                nn.Linear(512, 1, bias=True)
            )

    def forward(self, x):
        prediction = self.net(x)
        # prediction = prediction.view(-1, len(tags), 2)
        # prediction = F.argmax(prediction, dim=2)
        # prediction = prediction.squeeze(-1).float()
        return prediction


net = Net(column=COLUMN)
net = net.cuda()

crit = nn.BCEWithLogitsLoss().cuda()


def get_optimizer(net, lr=LR, freeze=FREEZE):
    if freeze == 0:
        lrs = []
        for i, param in enumerate(net.net.children()):
            lrs.append({'params': param.parameters()})
        lrs[-1] = {'params': param.parameters(), 'lr': LR * 10}
        optimizer = optim.Adamax(lrs, lr=lr, weight_decay=0.9)
    else:
        for i, param in enumerate(net.net.children()):
            param.requires_grad = False
            if i > FREEZE:
                break
        net.net.fc.requires_grad = True
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.9)
    return optimizer


net_before_parallel = net
net = nn.DataParallel(net)
if RESUME:
    net.load_state_dict(torch.load(RESUME))


def compute_fn(pair):
    x, y = pair
    x, y = x.cuda(), y.cuda().float()
    y = y.unsqueeze(1)
    prediction = net(x)
    loss = crit(prediction, y)
    # loss = Variable(loss, requires_grad=True)
    return prediction, loss


def compute_accuracy():
    logging.info("computing accuracy")
    count = 0
    total = 0
    for x, y in val_loader:
        x = x.cuda()
        total += x.shape[0]
        prediction = net(x).cpu().numpy()
        prediction = (prediction > 0.5).astype(np.int)
        target = np.expand_dims(y.detach().cpu().numpy().astype(np.int), axis=1)
        for i in range(y.shape[0]):
            if np.array_equal(prediction[i], target[i]):
                count += 1
    logging.info(f'\taccuracy = {count / total}')


def compute_accuracy_pair(prediction, label):
    count = 0
    prediction = prediction.detach().cpu().numpy()
    prediction = (prediction > 0.5).astype(np.int)
    target = np.expand_dims(label.detach().cpu().numpy().astype(np.int), axis=1)
    for i in range(label.shape[0]):
        if np.array_equal(prediction[i], np.array(target[i])):
            count += 1
    acc = count / prediction.shape[0]
    # logging.info(f'iter accuracy = {acc}')


# def compute_f2_pair(prediction, label):
#     prediction = prediction.detach().cpu().numpy()
#     prediction = (prediction > 0.5).astype(np.int)
#     target = label.detach().cpu().numpy().astype(np.int)
#     for i in range(label.shape[0]):
#         tp = np.sum(prediction[i] * target[i])
#         precision = tp / np.sum(target[i])
#         recall = tp /


if TRAIN:
    net = net.cuda()
    train_loader, val_loader = train_test_split(AmazonDataset(df, labels, column=COLUMN), batch_size=BS,
                                                random_seed=RANDOM_SEED)
    best_model = None
    for lr in (1e-4, 1e-5, 1e-6):
        logging.info(f'training with learning rate {lr}')
        optimizer = get_optimizer(net_before_parallel, lr=lr)
        if best_model is not None:
            logging.info(f"resume from {best_model}")
            net.load_state_dict(torch.load(best_model))
        best_model = train(net, train_loader, val_loader, optimizer, EPOCHS, compute_fn,
                           EarlyStopping(f'./models/amazon_best_{COLUMN}', patience=3),
                           epoch_end_callback=compute_accuracy)
else:
    # img = Image.open(IMAGE)
    # img = trfs(img)
    # img = img[:3, :, :]
    # img = img.unsqueeze(0)
    # with torch.no_grad():
    #     prediction = net(img)
    #     print(prediction)
    model_mapping = {'agriculture': './models/amazon_best_agriculture_0.19409638224692827',
                     'artisinal_mine': './models/amazon_best_artisinal_mine_0.014648641422686596',
                     'bare_ground': './models/amazon_best_bare_ground_0.02775007519390314',
                     'blooming': './models/amazon_best_blooming_0.017692260599384706',
                     'blow_down': './models/amazon_best_blow_down_0.012828032874802905',
                     'clear': './models/amazon_best_clear_0.11885210984710896',
                     'cloudy': './models/amazon_best_cloudy_0.03458153792501738',
                     'conventional_mine': './models/amazon_best_conventional_mine_0.01632822955423218',
                     'cultivation': './models/amazon_best_cultivation_0.09353890326038926',
                     'habitation': './models/amazon_best_habitation_0.06882091915106466',
                     'haze': './models/amazon_best_haze_0.05561411199199051',
                     'partly_cloudy': './models/amazon_best_partly_cloudy_0.054492494252582006',
                     'primary': './models/amazon_best_primary_0.0649511478999156',
                     'road': './models/amazon_best_road_0.10346179906571028',
                     'selective_logging': './models/amazon_best_selective_logging_0.017975151464934388',
                     'slash_burn': './models/amazon_best_slash_burn_0.016148214383671682',
                     'water': './models/amazon_best_water_0.14608877789802277'}
    # net.load_state_dict(torch.load(model_mapping['habitation']))
    result_frames = []
    for column, model in model_mapping.items():
        result_column = []
        net = Net(column=column)
        net = nn.DataParallel(net)
        print(f'working on column [{column}], loading model [{model}]')
        test_dataset = AmazonTestDataset()
        net.load_state_dict(torch.load(model))
        net = net.cuda()
        test_loader = DataLoader(test_dataset, batch_size=BS)
        results = []
        batch_count = 0
        with torch.no_grad():
            for im_fns, x in test_loader:
                batch_count += 1
                print(f'column {column}, batch count = {batch_count}')
                x = x.cuda()
                prediction = net(x).cpu().numpy()
                prediction = (prediction > 0.5).astype(np.int)
                for i in range(prediction.shape[0]):
                    result_column.append({'image_name': os.path.splitext(os.path.basename(im_fns[i]))[0],
                                          column: prediction[i][0]})
        result_df = pd.DataFrame(result_column, columns=['image_name', column])
        result_df.to_csv(f'submission_{column}.csv', index=False)
        print(f'done with column {column}')
    print('done')
