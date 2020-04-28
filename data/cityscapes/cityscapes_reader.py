import os
from os.path import join
import math
from operator import itemgetter
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import PIL.Image as pimg
import numpy as np

import utils
import data.transform as transform
import data.data_utils as data_utils
from data.cityscapes import labels


class_colors = [
    (128,64,128), (244,35,232), (70,70,70), (102,102,156), (190,153,153), (153,153,153),
    (250,170,30), (220,220,0), (107,142,35), (152,251,152), (70,130,180), (220,20,60), (255,0,0),
    (0,0,142), (0,0,70), (0,60,100), (0,80,100), (0,0,230), (119,11,32), (0,0,0)]
class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle']


class DatasetReader(Dataset):
    class_colors =  class_colors
    class_names =  class_names
    name_to_id_map = {}
    for i, class_name in enumerate(class_names):
        name_to_id_map[class_name] = i
    name = 'cityscapes'
    num_classes = 19
    num_logits = num_classes
    batchnorm_momentum = 0.05

    train_mapping, ignore_id = labels.get_train_mapping()

    submit_ids = labels.get_submit_ids()
    submit_mapping = np.zeros(num_classes, dtype=np.uint8)
    for i in range(len(submit_mapping)):
        submit_mapping[i] = submit_ids[i]

    category_map = labels.get_category_map()

    mean = np.array([73.45286, 83.21195, 72.78044], dtype=np.float32)
    std = np.array([45.19961, 46.3659, 45.48506], dtype=np.float32)


    def __init__(self, data_dir, subsets, args, batch_size, train=True, jitter=True):
        self.subsets = subsets
        self.batch_size = batch_size
        self.train = train
        self.jitter = jitter
        self.args = args
        self.model_downsampling_factor = args.downsampling_factor
        self.downsample = args.downsample
        self.multiscale_test = args.multiscale_test
        self.has_labels = not 'test' in subsets
        assert self.downsample == 0 or self.downsample == 2 or self.downsample == 4

        self.root_dir = data_dir
        img_dir = join(self.root_dir, 'leftImg8bit')
        labels_dir = join(self.root_dir, 'gtFine')

        if jitter:
            self.random_crop = args.random_crop
            self.crop_size = args.crop_size
            self.jitter_flip = args.jitter_flip
            self.jitter_scale = args.jitter_scale
            self.jitter_grayscale = False
        else:
            self.jitter_flip = False
            self.jitter_grayscale = False
            self.random_crop = False
            self.jitter_scale = False
        self.step = 0

        self.subsets = subsets

        self.img_paths = []
        self.label_paths = []
        self.names = []
        for subset in subsets:
            img_paths = []
            label_paths = []
            names = []
            subset_dir = join(img_dir, subset)
            cities = next(os.walk(subset_dir))[1]
            for city in cities:
                files = next(os.walk(join(subset_dir, city)))[2]
                img_paths.extend([join(subset_dir, city, f) for f in files])
                names.extend([f[:-4] for f in files])
                if self.has_labels:
                    suffix = '_gtFine_labelIds.png'
                    path_lst = [join(labels_dir, subset, city, f[:-16]+suffix) for f in files]
                    label_paths.extend(path_lst)

            self.img_paths.extend(img_paths)
            self.label_paths.extend(label_paths)
            self.names.extend(names)


        last_batch_size = len(self.names) % self.batch_size
        if train and last_batch_size > 0:
            data_utils.oversample_end([self.img_paths, self.label_paths, self.names],
                                      self.batch_size - last_batch_size)
            last_batch_size = len(self.names) % self.batch_size
            assert last_batch_size == 0 or last_batch_size == self.batch_size


        print('\nTotal num images =', len(self.names))
        print('Batch size =', self.batch_size)
        print('Last batch size =', last_batch_size, '\n')


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img = pimg.open(self.img_paths[idx])
        width, height = img.size
        if self.downsample:
            width //= self.downsample
            height //= self.downsample

        if self.has_labels:
            labels = pimg.open(self.label_paths[idx])

        batch = {}
        if self.multiscale_test:
            transform.multiscale_inference(img, self.mean, self.std, batch,
                                           self.model_downsampling_factor)

        if self.jitter_scale:
            scale = np.random.uniform(self.args.min_jitter_scale, self.args.max_jitter_scale)
            img_size = (round(scale * width), round(scale * height))
        else:
            img_size = (width, height)

        if not self.train:
            img_size = transform.pad_size_for_pooling(img_size, self.model_downsampling_factor)
        img = transform.resize_img(img, img_size)
        if self.has_labels:
            labels = transform.resize_labels(labels, img_size)
        if self.random_crop and max(img_size) > self.crop_size:
            img, labels = transform.random_crop([img, labels], self.crop_size, snap_margin_prob=0)
        if self.jitter_flip:
            img, labels = transform.random_flip([img, labels])
        if self.jitter_grayscale:
            img = transform.random_grayscale(img, prob=self.args.jitter_grayscale_prob)

        if self.has_labels:
            labels = np.array(labels, dtype=np.int64)
            labels = self.train_mapping[labels]

        img = transform.normalize(img, self.mean, self.std)

        if self.random_crop:
            if min(img.shape[:2]) < self.crop_size:
                target_size = (self.crop_size, self.crop_size)
                img = transform.pad(img, target_size, 0)
                labels = transform.pad(labels, target_size, self.ignore_id)

        img = transform.numpy_to_torch_image(img)
        batch['image'] = img
        batch['name'] = self.names[idx]
        if self.has_labels:
            labels = torch.from_numpy(labels)
            batch['labels'] = labels
            batch['target_size'] = labels.shape[:2]
        else:
            batch['target_size'] = [height, width]
        return batch


    @classmethod
    def denormalize(cls, img):
        return transform.denormalize(img, cls.mean, cls.std)

    @classmethod
    def map_to_submit_ids(cls, id_img, name):
        return cls.submit_mapping[id_img], name
