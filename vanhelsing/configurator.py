"""
This files help with dumping and loading config from argparse
"""
import argparse
import json
from argparse import Namespace
import os
import time
import sys


class MyArgumentParser(argparse.ArgumentParser):
    def _get_defaults(self):
        defaults = {}
        for action in self._positionals._actions:
            for st in action.option_strings:
                st = st.replace('--', '').replace('-', '')
                if action.type is not None:
                    defaults[st] = action.type(action.default)
                else:
                    defaults[st] = action.default
        return defaults

    def load(self, args, file_name):
        with open(file_name, 'r') as file:
            config = json.load(file)
            defaults = self._get_defaults()
            for key, value in args.__dict__.items():
                default_value = defaults[key]
                if value != default_value:
                    config[key] = value
            args.__dict__ = config

    @staticmethod
    def save(args, file_name):
        with open(file_name, 'w') as file:
            json.dump(args.__dict__, file)

    def parse_args(self, args=None, namespace=None) -> Namespace:
        self.add_argument('--config_file', help='location to config file', default=None)
        self.add_argument('--config_save', help='location to save config files', default=None)
        args = super().parse_args(args, namespace)
        save_current_config = args.config_save is not None
        if args.config_file is not None:
            print(f'loading config from file {args.config_file}')

            self.load(args, args.config_file)
            # do not consider config_save flag from previous run
            if not save_current_config:
                try:
                    del args.config_save
                except:
                    pass
        print(f'current configs {args}')
        if 'config_save' in args and args.config_save is not None:
            config_save = args.config_save
            del args.config_save
            print(f'saving config to folder {config_save}')
            if not os.path.exists(config_save):
                os.mkdir(config_save)
            config_fn = f"config-{sys.argv[0]}-{time.strftime('%Y%m%d_%H%M%S')}.json"
            config_fn = os.path.join(config_save, config_fn)
            print(f'configs are going to be saved in {config_fn}')
            MyArgumentParser.save(args, config_fn)
        return args


class TrainArgumentParser(MyArgumentParser):
    def parse_args(self, args=None, namespace=None) -> Namespace:
        self.add_argument('--resume', help='location to resume from')
        self.add_argument('--resume_epoch', default=0, type=int, help='epoch to resume from')
        self.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        self.add_argument('--bs', default=32, type=int, help='batch size')
        self.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
        self.add_argument('--debug_iter', default=100, type=int, help='number of iteration between every debug message')
        self.add_argument('--random_seed', default=42, type=int, help='number of epochs to train')
        self.add_argument('--split_ratio', default=.1, type=float, help='train/test split')
        self.add_argument('--patience', default=15, type=int, help='patience for early stopping')
        args = super().parse_args(args, namespace)
        return args
