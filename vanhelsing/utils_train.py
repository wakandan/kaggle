import torch
import numpy as np
from torch.optim import Optimizer
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from vanhelsing.logger import *
from typing import *


class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""

    def __init__(self, save_location, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_location = save_location
        self.best_saved_location = save_location

    def __call__(self, val_loss, model):

        score = - val_loss

        if self.best_score is None:
            self.best_score = score
            logging.info(f"saving the first model val_loss = {val_loss}")
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            logging.warning(
                'EarlyStopping counter: {} out of {}, last val_loss = {}'.format(self.counter, self.patience,
                                                                                 self.val_loss_min))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info('EarlyStopping counter reset, best_score = {}'.format(val_loss))
            self.best_score = score
            self.best_saved_location = self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        :param val_loss:
        :param model:
        :return: location to the best model file
        """
        save_location = self.save_location + '_' + str(val_loss)
        logging.info(
            'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model to {}'.format(self.val_loss_min, val_loss,
                                                                                        save_location))
        torch.save(model.state_dict(), save_location)

        self.val_loss_min = val_loss
        return save_location


def train_test_split(dataset: Dataset, split=.2, random_seed=42, batch_size=32):
    """
    Split a dataset into train/test loaders
    :param dataset:
    :param split: float
    :param random_seed:
    :param batch_size:
    :return: train dataloader, test dataloader
    """
    np.random.seed(random_seed)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split_index = int(np.floor(len(dataset) * split))
    train_idx, val_idx = indices[split_index:], indices[:split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size)
    return train_loader, val_loader


def train(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, epochs,
          compute_fn: Callable,
          early_stopping: EarlyStopping, epoch_end_callback: Callable = None, debug_step=100,
          iter_end_callback: Callable = None, scheduler=None):
    """
    :param net:
    :param train_loader:
    :param val_loader:
    :param optimizer:
    :param epochs:
    :param compute_fn: return loss value
    :param early_stopping:
    :param debug_step:
    :param scheduler: should be a cosine scheduler
    :return:
    """
    stopped_training = False
    for e in range(epochs):
        scheduler.step()
        net.train(True)
        running_loss = []
        for i, pair in enumerate(train_loader):
            prediction, loss = compute_fn(pair)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
            if i % debug_step == 0:
                logging.info(f'epoch = {e}, iter = {i}, train loss = {np.average(running_loss)}')
                running_loss = []  # look at train loss for each batch
                if iter_end_callback is not None:
                    iter_end_callback(prediction, pair[1])
        with torch.no_grad():
            running_loss = []
            for i, pair in enumerate(val_loader):
                prediction, loss = compute_fn(pair)
                running_loss.append(loss.item())
                if i % debug_step == 0:
                    logging.info(f'epoch = {e}, iter = {i}, val loss = {np.average(running_loss)}')
            if epoch_end_callback is not None:
                epoch_end_callback()
            early_stopping(np.average(running_loss), net)
            if early_stopping.early_stop:
                logging.info("early stopped")
                stopped_training = True
                break
        if stopped_training:
            break
    return early_stopping.best_saved_location


class Identity(nn.Module):
    """
    Identity module that returns the input. Use this to bypass layers in trasfer learning
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_lr(optimizer: Optimizer):
    """
    Get the current learning rate used by optimizer
    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
