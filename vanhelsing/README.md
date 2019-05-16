# Vanhelsing

Vanhelsing is a repo to collect common best practices used during in vision team

## Training

### Best practices with ArgumentParser

Most training code will have common settings such as `learning_rate`, `batch_size` and so on. Therefore, vanhel has implemented a couple of classes to simplify the life of our researchers

`MyArgumentParser` extends the default `ArgumentParser` with 2 flags 

- `--config_save <folder>` to specify where the config files for each run will be stored
- `--config_file <file>` to specify the config file to load from

`TrainArgumentParser` extends `MyArgumentParser` with extra flags:
- `--resume <model_file>` resume training from a model file
- `--resume_epoch <int>` resume from an epoch. This simply increase the epoch count so that the model files are stored correctly
- `--lr <float>` learning rate
- `--bs <int>` batch size
- `--epochs <int>` num of epochs
- `--debug_iter <int>` how many iteration between prints
- `--random_seed <int>` random seed for np, torch and the like. Should always have this. Default = 42
- `--split_ratio <float>` train/test split ratio
- `--patience <int>` patience for early stopping

In your code, you should extend from `TrainArgumentParser` and add more. Later, it's recommended to define a variable for each flag to aid code auto-complete

```python
parser = TrainArgumentParser()

parser.add_argument('--input', help='training csv file')
parser.add_argument('--freeze', help='freeze base net', default=False)
parser.add_argument('--train', help='train', action='store_true')
...

args = parser.parse_args()

RESUME = args.resume
RESUME_EPOCH = args.resume_epoch
LR = args.lr
BS = args.bs
EPOCHS = args.epochs
DEBUG_ITER = args.debug_iter
RANDOM_SEED = args.random_seed
SPLIT_RATIO = args.split_ratio
PATIENCE = args.patience
...
```  

### Early stopping (pytorch)

Implement early stopping with simple checking for validation loss 

```python
from vanhelsing.utils_train import *

early_stopping = EarlyStopping('./models/angle_prediction_best.pt', patience=5, verbose=True)
for e in range(epochs):
    train(...)
    val_loss = val(...)
    early_stopping(val_loss, net)
    if early_stopping.early_stop:
        logging.error('early stopping')
        stop_training()
```

### Cosine annealing LR with restart

Code from here https://github.com/mpyrozhok/adamwr

```python
from vanhelsing.cosine_scheduler import CosineLRWithRestarts
from vanhelsing.adamw import AdamW

model = resnet()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=5, t_mult=1.2)
for epoch in range(100):
    scheduler.step()
    train_for_every_batch(...)
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.batch_step()
    validate(...)
``` 