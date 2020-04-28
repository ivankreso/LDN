import os
import sys
from os.path import join, isfile, dirname
import argparse
import time
import shutil
from multiprocessing import Pool
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import PIL.Image as pimg
from tqdm import tqdm

import utils
import evaluation
import libs.cylib as cylib


def save_mat_to_img(mat, save_path):
    min_val = mat.min()
    max_val = mat.max()
    normalized = (mat - min_val) / (max_val - min_val)
    img = (normalized * 255).astype(np.uint8)
    img = pimg.fromarray(img)
    saver_pool.apply_async(img.save, [save_path])


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--params-version', type=str, default='params_best')
parser.add_argument('--depth', type=int, default=121)
parser.add_argument('--reader', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--pretrained', type=int, default=1)
parser.add_argument('--subsets', type=str, default='val') # train, val, test
parser.add_argument('--downsample', type=int, default=0)
parser.add_argument('--checkpointing', type=int, default=0)
parser.add_argument('--multiscale-test', type=int, default=0)
parser.add_argument('--save-outputs', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--aux-loss', type=int, default=0)
parser.add_argument('--save-submit', type=int, default=0)
args = parser.parse_args()

reader_path = args.reader
reader = utils.import_module('reader', reader_path)
class_colors = reader.DatasetReader.class_colors
class_names = reader.DatasetReader.class_names
ignore_id = reader.DatasetReader.ignore_id

print(args.model, args.depth)
net_model = utils.import_module('net_model', args.model)
model = net_model.build(depth=args.depth, pretrained=args.pretrained,
                        dataset=reader.DatasetReader, args=args)
state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.cuda()
model.eval()

subsets = args.subsets.split(',')

dataset = reader.DatasetReader(args.dataset, subsets, args, args.batch_size, train=False, jitter=False)

data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=6, pin_memory=True, drop_last=False)

save_dir = join('outputs', dataset.name, '_'.join(subsets))
if args.save_outputs and os.path.exists(save_dir):
    shutil.rmtree(save_dir)
    save_color = join(save_dir, 'color')
    os.makedirs(save_color, exist_ok=True)
    os.makedirs(join(save_dir, 'submit'), exist_ok=True)

if args.save_outputs or args.save_submit:
    saver_pool = Pool(processes=4)

if dataset.has_labels:
    conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)

log_interval = max(1, len(data_loader) // 32)

for step, batch in enumerate(tqdm(data_loader)):
    if dataset.has_labels:
        labels = batch['labels']
        true = labels.numpy().astype(np.int32)

    target_size = batch['target_size']
    target_size = [val[0] for val in target_size]
    if dataset.multiscale_test:
        pred = evaluation.multiscale_forward(batch, model, target_size)
    else:
        pred = evaluation.forward(batch, model, args, target_size=target_size,
                                    return_aux=args.aux_loss)
        if args.aux_loss:
            pred, aux_preds = pred

    pred = pred.numpy().astype(np.int32)
    if dataset.has_labels:
        cylib.collect_confusion_matrix(pred.reshape(-1), true.reshape(-1), conf_mat)

    pred = pred[0]
    name = batch['name'][0]
    if dataset.has_labels:
        true = true[0]

    if args.save_submit:
        pred = pred.astype(np.uint8)
        submit_img, submit_name = dataset.map_to_submit_ids(pred, name)
        submit_img = pimg.fromarray(submit_img)
        save_path = join(save_dir, 'submit', '%s.png' % submit_name)
        saver_pool.apply_async(submit_img.save, [save_path])

    if args.save_outputs:
        img_pred = evaluation.colorize_labels(pred, class_colors)
        if args.multiscale_test:
            img_raw = dataset.denormalize(batch['multiscale_imgs'][2][0])
        else:
            img_raw = dataset.denormalize(batch['image'][0])
        if dataset.has_labels:
            img_true = evaluation.colorize_labels(true, class_colors)
            img_errors = img_pred.copy()
            correct_mask = pred == true
            ignore_mask = true == ignore_id
            img_errors[correct_mask] = 0
            img_errors[ignore_mask] = 0
            num_mistakes = (pred != true).sum() - ignore_mask.sum()
            img1 = np.concatenate((img_raw, img_true), axis=1)
            img2 = np.concatenate((img_errors, img_pred), axis=1)
            img = np.concatenate((img1, img2), axis=0)
            filename = '%06d_%s.jpg' % (num_mistakes, name)
            save_path = join(save_color, filename)
        else:
            img_raw = transform.resize_img(pimg.fromarray(img_raw), reversed(img_pred.shape[:2]))
            img_raw = np.array(img_raw, dtype=np.uint8)
            img = np.concatenate((img_raw, img_pred), axis=0)
            save_path = join(save_color, '%s.jpg' % (name))


if dataset.has_labels:
    print('')
    pixel_acc, miou, ciou, recall, precision, _ = evaluation.get_eval_metrics(
            conf_mat, 'Validation', class_names, verbose=True)
    if dataset.category_map is not None:
        evaluation.get_category_iou(conf_mat, dataset.category_map)
    # plot_utils.plot_confusion_matrix(conf_mat, class_names)
