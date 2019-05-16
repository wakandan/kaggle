import glob
import sys

# torch.backends.cudnn.benchmark = True

sys.path.append('..')

from torch import optim
import pandas as pd
from pathlib import Path
from torchvision import models, transforms
import random
from PIL import Image
from vanhelsing.utils_train import *
from vanhelsing.configurator import *
from sklearn.metrics import fbeta_score

parser = TrainArgumentParser()

parser.add_argument('--train', action='store_true')
parser.add_argument('--model')
parser.add_argument('--image')
parser.add_argument('--freeze', type=int, default=0)

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

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f'CSV location: {CSV}')
df = pd.read_csv(CSV)
df['tag_list'] = df['tags'].apply(lambda x: x.split())
df['image_name'] = df['image_name'].apply(lambda x: x + '.jpg')
fns = list(df['image_name'])
tags = sorted(list(set([i for sublist in list(df['tag_list']) for i in sublist])))
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
    # transforms.RandomRotation(90),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        im_fn = row['image_name']
        im = Image.open(ROOT / f'train-jpg/{im_fn}')
        im = trfs(im)
        im = im[:3, :, :]
        lbl = torch.from_numpy(labels[index])
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

    def __init__(self):
        super().__init__()
        self.net = models.resnet34(pretrained=True)
        self.net.fc = nn.Sequential(
            # nn.Linear(512, 256, bias=True),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, len(tags)),
        )

    def forward(self, x):
        prediction = self.net(x)
        # prediction = prediction.view(-1, len(tags), 2)
        # prediction = F.argmax(prediction, dim=2)
        # prediction = prediction.squeeze(-1).float()
        return prediction


net = Net().cuda()

crit = nn.BCEWithLogitsLoss()
# if FREEZE == 0:
#     lrs = []
#     for i, param in enumerate(net.net.children()):
#         lrs.append({'params': param.parameters()})
#     lrs[-1] = {'params': param.parameters(), 'lr': LR * 10}
#     optimizer = optim.Adamax(lrs, lr=LR, weight_decay=0.9)
# else:
#     for i, param in enumerate(net.net.children()):
#         param.requires_grad = False
#         if i > FREEZE:
#             break
#     net.net.fc.requires_grad = True
#     optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.9)
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.9)

# net = nn.DataParallel(net)
if RESUME:
    net.load_state_dict(torch.load(RESUME))


def compute_fn(pair):
    x, y = pair
    x, y = x.cuda(), y.cuda().float()
    prediction = net(x)
    # loss = crit(y, prediction)
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
        for i in range(y.shape[0]):
            if np.array_equal(prediction[i], y[i]):
                count += 1
    logging.info(f'\taccuracy = {count / total}')


def compute_f2():
    logging.info('computing f2 score')
    f2s = []
    for x, y in val_loader:
        x = x.cuda()
        prediction = net(x).cpu().numpy()
        prediction = (prediction > 0.5).astype(np.int)
        for i in range(y.shape[0]):
            f2 = fbeta_score(y[i], prediction[i], beta=2, average='micro')
            f2s.append(f2)
    logging.info(f'f2 score = {np.mean(np.asarray(f2s))}')


if TRAIN:
    # net = net.cuda()
    train_loader, val_loader = train_test_split(AmazonDataset(df), batch_size=BS, random_seed=RANDOM_SEED)
    train(net, train_loader, val_loader, optimizer, EPOCHS, compute_fn=compute_fn,
          early_stopping=EarlyStopping('./models/amazon_best', patience=5),
          epoch_end_callback=compute_f2,
          debug_step=args.debug_iter)
else:
    # img = Image.open(IMAGE)
    # img = trfs(img)
    # img = img[:3, :, :]
    # img = img.unsqueeze(0)
    # with torch.no_grad():
    #     prediction = net(img)
    #     print(prediction)
    test_dataset = AmazonTestDataset()
    net.load_state_dict(torch.load(MODEL))
    test_loader = DataLoader(test_dataset, batch_size=32)
    results = []
    with torch.no_grad():
        for im_fns, x in test_loader:
            x = x.cuda()
            prediction = net(x).cpu().numpy()
            prediction = (prediction > 0.5).astype(np.int)
            for i in range(prediction.shape[0]):
                results.append({'image_name': os.path.splitext(os.path.basename(im_fns[i]))[0],
                                'tags': ' '.join(decode_tags(prediction[i]))})
    result_df = pd.DataFrame(results, columns=['image_name', 'tags'])
    print(result_df.head())
    result_df.to_csv(f'submission.csv', index=False)
    print('done')
