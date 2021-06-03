from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn as nn
import numpy as np
import time
from utils import common_util
from torch.utils.data import DataLoader
from eval import eval_utils

from data_processing.timit import TIMIT, variable_collate_fn
from model.CTCNetwork import ConnectionistTemporalClassification
import argparse

is_cuda = torch.cuda.is_available()

def get_optimizer(model):
    optimizer = None
    scheduler = None
    return optimizer, scheduler

def train(train_generator, test_generator, output_root_dir, model):
    if is_cuda:
        model = model.cuda()

    optimizer, scheduler = get_optimizer(model)

    best_ler = eval_utils.evaluate(test_generator, model, is_cuda)

    exp_loss = None
    global_step = 0

    model.zero_grad()
    print_interval = 1000 * model.model_config.batch_size

    for epoch in range(model.model_config.num_epoch):
        optimizer.zero_grad()
        for inputs in train_generator:
            model.train()
            if is_cuda:
                for k, v in inputs.items():
                    inputs[k] = v.cuda()

            logits = model(**inputs)
            loss_t = model.get_loss(logits, **inputs)

            loss_t.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model.model_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            exp_loss = 0.99 * exp_loss + 0.01 * loss_t.item() if exp_loss else loss_t.item()

            if global_step > 0 and global_step % print_interval == 0:
                print(f'{global_step} / {t_total} train loss: {exp_loss} lr: {scheduler.get_lr()[0]}', flush=True)

        ler = eval_utils.evaluate(test_generator, model, is_cuda)
        if best_ler > ler:
            best_ler = ler
            output_dir = os.path.join(output_root_dir, 'checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"Saving model checkpoint to {output_dir}", flush=True)

            common_util.save_pretrained(model, output_dir)

    print(f'{global_step} / {t_total} train loss: {exp_loss} lr: {scheduler.get_lr()[0]}', flush=True)

def process_train(args):
    model_config = common_util.get_config(args.config_file)

    common_util.set_seed(model_config.seed, is_cuda)

    train_dataset = TIMIT(os.path.join(args.data_dir, 'TRAIN'))
    train_dataset.dump_mean_var(args.model_dir)
    train_dataset.dump_phone_vocab(args.model_dir)

    test_dataset = TIMIT(os.path.join(args.data_dir, 'TEST'))

    output_root_dir = os.path.join(args.model_dir, f'dl_model_{int(time.time())}')
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    print(f'model out dir {output_root_dir}', flush=True)

    train_dataset.init_dataset(args.model_dir)
    test_dataset.init_dataset(args.model_dir)

    train_generator = DataLoader(train_dataset,
                                 batch_size=model_config.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=variable_collate_fn)
    test_generator = DataLoader(test_dataset,
                                 batch_size=model_config.batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=variable_collate_fn)

    model = ConnectionistTemporalClassification(model_config)

    train(train_generator, test_generator, output_root_dir, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-d",
                        dest="data_dir",
                        required=True,
                        default=None,
                        help="Data dir.")
    parser.add_argument("-m",
                        dest="model_dir",
                        required=True,
                        default=None,
                        help="Model root dir.")
    parser.add_argument("-c",
                        dest="config_file",
                        required=True,
                        default=None,
                        help="Model config file.")
    args = parser.parse_args()

    process_train(args)
