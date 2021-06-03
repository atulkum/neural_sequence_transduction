from __future__ import absolute_import, division, print_function

import os.path

import os
import torch
import torch.nn as nn
import numpy as np
import random
import time
import json
import codecs
from data_processing.timit import TIMIT, get_dataloader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from model.CTCNetwork import ConnectionistTemporalClassification
import common_util
from eval import eval_utils

is_cuda = torch.cuda.is_available()

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_optimizer(model, t_total):
    optimizer = SGD(model.parameters(),
                    lr = model.model_config.lr,
                    momentum=model.model_config.momentum)
    scheduler = None #get_linear_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=model.model_config.warmup_steps,
    #                                            num_training_steps=t_total)
    return optimizer, scheduler

def train(output_root_dir, train_data, dev_data, model_config):
    model = ConnectionistTemporalClassification(model_config)
    if is_cuda:
        model = model.cuda()

    data_size = len(train_data)
    num_batch = np.ceil(data_size / model_config.batch_size)
    t_total = model_config.num_epoch * num_batch
    optimizer, scheduler = get_optimizer(model, t_total)

    exp_loss = None
    global_step = 0
    best_score = eval_utils.evaluate(dev_data, model, model_config.batch_size, is_cuda)
    print_interval = 1000 * model_config.batch_size

    for epoch in range(model_config.num_epoch):
        dataloader = get_dataloader(train_data, model_config.batch_size, True)
        model.train()
        for i_batch, inputs in enumerate(dataloader):
            if is_cuda:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()

            optimizer.zero_grad()
            logprobs = model(**inputs)
            loss = model.get_loss(logprobs, inputs['phone'], inputs['length'])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)
            optimizer.step()
            #scheduler.step()

            global_step += 1

            exp_loss = 0.99 * exp_loss + 0.01 * loss.item() if exp_loss else loss.item()

            if True: #global_step > 0 and global_step % print_interval == 0:
                #print(f'{global_step} / {t_total} train loss: {exp_loss} lr: {scheduler.get_lr()[0]}', flush=True)
                print(f'{global_step} / {t_total} train loss: {exp_loss}', flush=True)

        ler = eval_utils.evaluate(dev_data, model, model_config.batch_size, is_cuda)
        print(f'{global_step}/{t_total} LER {ler:.5f}', flush=True)

        if ler < best_score :
            best_score = ler
            # output_dir = os.path.join(output_root_dir, 'checkpoint-{}'.format(epoch))
            output_dir = os.path.join(output_root_dir, 'checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"Saving model checkpoint to {output_dir}", flush=True)

            model.save_pretrained(output_dir)

            with open(os.path.join(output_dir, "training_config.json"), 'w') as fout:
                json.dump(vars(model_config), fout)

    #print(f'{global_step} / {t_total} train loss: {exp_loss} lr: {scheduler.get_lr()[0]}', flush=True)
    print(f'{global_step} / {t_total} train loss: {exp_loss}', flush=True)

def process_train(root_dir, data_dir, model_config_filename):
    output_root_dir = os.path.join(root_dir, 'dl_model') #f'dl_model_{int(time.time())}')
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    print(f'model out dir {output_root_dir}', flush=True)

    train_data_dir = os.path.join(data_dir, 'TRAIN')
    train_dataset = TIMIT('train', train_data_dir, train_data_dir, False)
    train_dataset.load_stat_vocab(train_data_dir)

    test_data_dir = os.path.join(data_dir, 'TEST')
    dev_dataset = TIMIT('test', test_data_dir, test_data_dir, False)
    dev_dataset.load_stat_vocab(train_data_dir)

    model_config = common_util.get_config(model_config_filename)
    model_config.num_tags = len(train_dataset._phone_vocab)
    train(output_root_dir, train_dataset, dev_dataset, model_config)

if __name__ == "__main__":
    root_dir = '/Users/atulkumar/neural_sequence_transduction/models/'
    data_dir = '/Users/atulkumar/neural_sequence_transduction/TIMIT'
    model_config_filename = '/Users/atulkumar/neural_sequence_transduction/config/base-model.json'
    process_train(root_dir, data_dir, model_config_filename)