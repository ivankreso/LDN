import os
import shutil
import time
import importlib.util
from datetime import datetime
from math import log10

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_num_params(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
    return num_params


def get_expired_time(start_time):
    curr_time = time.perf_counter()
    delta = curr_time - start_time
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds


def get_eta_time(start_time, curr_iter, num_iters):
    curr_time = time.perf_counter()
    delta = curr_time - start_time
    eta = ((num_iters - curr_iter) / curr_iter) * delta

    hour = int(eta / 3600)
    eta -= hour * 3600
    minute = int(eta / 60)
    eta -= minute * 60
    seconds = eta
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds


def get_time_string():
    time = datetime.now()
    name = str(time.year) + f"_{time.month:02d}" + f"_{time.day:02d}" \
        + f"_{time.hour:02d}" + f"{time.minute:02d}" + f"{time.second:02d}"
    return name


def get_time():
    time = datetime.now()
    return '%02d' % time.hour + ':%02d' % time.minute + ':%02d' % time.second


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            # finetuning is unstable if we don't also freeze params
            module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()


def set_train_mode(model, freeze_bachnorm):
    model.train()

    if not freeze_bachnorm:
        unfreeze_batch_norm(model)
    else:
        freeze_batch_norm(model)


def set_learning_rate_linear(optimizer, step, num_steps, lr_min, lr_max):
    loglr = (log10(lr_max) - log10(lr_min)) * (1 - step / num_steps) + log10(lr_min)
    lr = 10**loglr
    for param_group in optimizer.param_groups:
        if 'lr_factor' in param_group:
            param_group['lr'] = lr / param_group['lr_factor']
        else:
            param_group['lr'] = lr
    return lr


# from bokeh.io import output_file, show, save
# from bokeh.layouts import column
# from bokeh.plotting import figure
# def lr_find(model, data_loader, optimizer, lr_min=1e-6, lr_max=1, num_iters=256):
def lr_find(model, data_loader, optimizer, lr_min=1e-6, lr_max=1, num_iters=512):
    step = 0
    done = False
    final_step = num_iters - 1
    lr_data = []
    loss_data = []
    while not done:
        for batch in data_loader:
            if step == num_iters:
                done = True
                break
            optimizer.zero_grad()
            model.send_to_gpu(batch)
            loss = model.forward_loss(batch)
            loss.backward()
            optimizer.step()
            lr = set_learning_rate_linear(optimizer, final_step-step, final_step, lr_min, lr_max)
            lr_data.append(lr)
            loss_data.append(loss.item())
            print(step, lr, '->', loss.item())
            step += 1
    # plot data
    # fig = figure(plot_width=1024, plot_height=512, title='loss/lr')
    fig = figure(plot_width=1024, plot_height=512, x_axis_type='log', title='loss/lr')
    fig.line(x=lr_data, y=loss_data, line_width=4, color='navy')
    output_file('/tmp/plot.html')
    save(fig)


def release_memory(model):
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1,3,32,32).cuda())
    model.train()
    # torch.cuda.empty_cache()
    # gc.collect()


def print_memory_stats(msg=''):
    alloc_size = torch.cuda.memory_allocated() / 1024**2
    max_alloc_size = torch.cuda.max_memory_allocated() / 1024**2
    cache_size = torch.cuda.memory_cached() / 1024**2
    max_cache_size = torch.cuda.max_memory_cached() / 1024**2
    print(msg)
    print('ALLOCATED =', alloc_size, max_alloc_size)
    print('CACHED =', cache_size, max_cache_size)
    print()


def mkdir(path, clean=False):
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)


def map_ids_to_classes(class_ids, class_names):
    names = []
    for cid in class_ids:
        if cid < len(class_names):
            names.append(class_names[cid])
    return names