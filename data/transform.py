import torch
import math
import random

import PIL.Image as pimg
from PIL import ImageFilter
import numpy as np


def resize_img(img, size, mode=pimg.BICUBIC):
    if img.size != size:
        return img.resize(size, pimg.BICUBIC)
    return img


def resize_labels(img, size):
    if img.size != size:
        img = img.resize(size, pimg.NEAREST)
    return img


def numpy_to_torch_image(img):
    img = torch.from_numpy(img)
    return img.permute(2,0,1).contiguous()


def _sample_location(dim_size, crop_size, snap_margin_prob):
    if dim_size > crop_size:
        max_start = dim_size - crop_size
        snap_margin = int(snap_margin_prob/2 * max_start)
        start_pos = np.random.randint(-snap_margin, max_start+1+snap_margin)
        start_pos = max(start_pos, 0)
        start_pos = min(start_pos, max_start)
        size = crop_size
    else:
        start_pos = 0
        size = dim_size
    return start_pos, size


def random_rotate(img, labels, ignore_id, max_angle=20, rgb_mean=None):
    angle = 0
    while angle == 0:
        angle = np.random.randint(-max_angle, max_angle+1)
    img = img.rotate(angle, resample=pimg.BICUBIC, expand=0)

    labels = np.array(labels, dtype=np.uint8)
    labels[labels==0] = 255
    labels = pimg.fromarray(labels).rotate(angle, resample=pimg.NEAREST, expand=0)
    labels = np.array(labels, dtype=np.uint8)
    labels[labels==0] = ignore_id
    labels[labels==255] = 0
    labels = pimg.fromarray(labels)
    return img, labels


def random_crop(images, crop_size, snap_margin_prob=0.1):
    if isinstance(crop_size, int):
        crop_h = crop_size
        crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    height = images[0].size[1]
    width = images[0].size[0]

    sx, crop_w = _sample_location(width, crop_w, snap_margin_prob)
    sy, crop_h = _sample_location(height, crop_h, snap_margin_prob)

    cropped = []
    for img in images:
        cropped.append(img.crop((sx, sy, sx+crop_w, sy+crop_h)))
    return cropped


def random_flip(images):
    if np.random.choice(2):
        flipped = []
        for img in images:
            img = img.transpose(pimg.FLIP_LEFT_RIGHT)
            flipped.append(img)
        return flipped
    else:
        return images

def random_grayscale(image, prob):
    s = np.random.uniform(0, 1)
    if s <= prob:
        image = image.convert('L').convert('RGB')
    return image


def random_blur(image, prob):
    s = np.random.uniform(0, 1)
    if s <= prob:
        return image.filter(ImageFilter.GaussianBlur(1))
    return image


def get_size_for_inference(size, up_size):
    larger_dim = np.argmax(size)
    smaller_dim = larger_dim ^ 1
    new_size = [0, 0]
    new_size[larger_dim] = up_size
    ar = size[smaller_dim] / size[larger_dim]
    new_size[smaller_dim] = round(new_size[larger_dim] * ar)
    return self.pad_size_for_pooling(new_size)


def pad(img, target_size, value):
    height = img.shape[0]
    width = img.shape[1]
    new_shape = list(img.shape)
    new_shape[0] = target_size[0]
    new_shape[1] = target_size[1]
    padded_img = np.ndarray(new_shape, dtype=img.dtype)
    padded_img.fill(value)
    sh = round((target_size[0] - height) / 2)
    eh = sh + height
    sw = round((target_size[1] - width) / 2)
    ew = sw + width
    padded_img[sh:eh,sw:ew,...] = img
    return padded_img


def normalize(img, mean, std):
    img = np.array(img, dtype=np.float32)
    img -= mean
    img /= std
    return img


def denormalize(img, mean, std):
    img = img.cpu().permute(1,2,0).contiguous().numpy()
    img *= std
    img += mean
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype(np.uint8)
    return img


def build_pyramid_labels(labels, pyramid_loss_scales, num_classes, batch):
    aux_target_dist = []
    aux_valid_mask = []
    for f in pyramid_loss_scales:
        target_dist, valid_mask = downsample_labels(labels, f, num_classes)
        aux_target_dist.append(target_dist)
        aux_valid_mask.append(valid_mask)
    batch['aux_targets'] = aux_target_dist
    batch['aux_valid_mask'] = aux_valid_mask


def build_pyramid_labels_th(batch, pyramid_loss_scales, num_classes):
    with torch.no_grad():
        labels = batch['labels']
        aux_target_dist = []
        aux_valid_mask = []
        for f in pyramid_loss_scales:
            target_dist, valid_mask = downsample_labels_th(labels, f, num_classes)
            aux_target_dist.append(target_dist)
            aux_valid_mask.append(valid_mask)
        batch['aux_targets'] = aux_target_dist
        batch['aux_valid_mask'] = aux_valid_mask


def downsample_labels_nearest(labels, factor, num_classes, to_torch=True):
    h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    img = pimg.fromarray(labels.astype(np.uint8))
    img = img.resize((new_w, new_h), pimg.NEAREST)
    labels = np.array(img, dtype=np.int64)
    return torch.from_numpy(labels), 0


def downsample_labels(labels, factor, num_classes, to_torch=True):
    h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.reshape(new_h, factor, new_w, factor)
    # +1 class here because ignore id = num_classes
    labels_oh = np.eye(num_classes+1, dtype=np.float32)[labels_4d]
    target_dist = labels_oh.sum((1, 3)) / factor**2
    C = target_dist.shape[-1]
    target_dist = target_dist.reshape(-1, C)
    N = target_dist.shape[0]
    # keep only boxes which have p(ignore) < 0.5
    valid_mask = target_dist[:,-1] < 0.5
    target_dist = np.ascontiguousarray(target_dist[:,:-1])
    dist_sum = target_dist.sum(1, keepdims=True)
    # avoid division by zero
    dist_sum[dist_sum==0] = 1
    # renormalize distribution after removing p(ignore)
    target_dist /= dist_sum
    if to_torch:
        target_dist = torch.from_numpy(target_dist)
        valid_mask = torch.from_numpy(valid_mask.astype(np.uint8))
    return target_dist, valid_mask


def downsample_distribution(labels, factor, num_classes):
    h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = np.ascontiguousarray(labels.reshape(new_h, factor, new_w, factor), labels.dtype)
    labels_oh = np.eye(num_classes, dtype=np.float32)[labels_4d]
    target_dist = labels_oh.sum((1, 3)) / factor ** 2
    return target_dist


def downsample_distribution_th(labels, factor, num_classes, ignore_id=None):
    n, h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.view(n, new_h, factor, new_w, factor)
    labels_oh = torch.eye(num_classes).to(labels_4d.device)[labels_4d]
    target_dist = labels_oh.sum(2).sum(3) / factor ** 2
    return target_dist


def downsample_labels_th(labels, factor, num_classes):
    '''
    :param labels: Tensor(N, H, W)
    :param factor: int
    :param num_classes:  int
    :return: FloatTensor(-1, num_classes), ByteTensor(-1, 1)
    '''
    n, h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.view(n, new_h, factor, new_w, factor)
    # +1 class here because ignore id = num_classes
    labels_oh = torch.eye(num_classes + 1).to(labels_4d.device)[labels_4d]
    target_dist = labels_oh.sum(2).sum(3) / factor ** 2
    C = target_dist.shape[-1]
    target_dist = target_dist.view(-1, C)
    # keep only boxes which have p(ignore) < 0.5
    valid_mask = target_dist[:, -1] < 0.5
    target_dist = target_dist[:, :-1].contiguous()
    dist_sum = target_dist.sum(1, keepdim=True)
    # avoid division by zero
    dist_sum[dist_sum == 0] = 1
    # renormalize distribution after removing p(ignore)
    target_dist /= dist_sum
    return target_dist, valid_mask


def pad_size_for_pooling(size, last_block_pooling):
    new_size = list(size)
    for i in range(len(new_size)):
        mod = new_size[i] % last_block_pooling
        if mod > 0:
            new_size[i] += last_block_pooling - mod
    return tuple(new_size)


def fit_size_for_pooling(size, last_block_pooling):
    new_size = list(size)
    for i in range(len(new_size)):
        mod = new_size[i] % last_block_pooling
        if mod > 0:
            if np.random.choice(2):
                new_size[i] += last_block_pooling - mod
            else:
                new_size[i] -= mod
    return tuple(new_size)


def multiscale_inference(img, mean, std, batch, last_block_pooling):
    scales = [0.5, 0.75, 1, 1.5, 2]     # 78.5 -> 80.07 (noflip 79.91)
    img_pyr = _build_pyramid(img, scales, mean, std, last_block_pooling)
    img_flip = img.transpose(pimg.FLIP_LEFT_RIGHT)
    img_pyr_flip = _build_pyramid(img_flip, scales, mean, std, last_block_pooling)
    batch['multiscale_imgs'] = img_pyr
    batch['multiscale_imgs_flip'] = img_pyr_flip
    return batch


def _build_pyramid(img, scales, mean, std, last_block_pooling):
    width, height = img.size
    img_pyr = []
    for s in scales:
        img_lvl = img
        img_size = (round(s * width), round(s * height))
        img_size = pad_size_for_pooling(img_size, last_block_pooling)
        img_lvl = resize_img(img_lvl, img_size)
        img_lvl = np.array(img_lvl, dtype=np.float32)
        img_lvl -= mean
        img_lvl /= std
        img_lvl = torch.from_numpy(img_lvl)
        img_lvl = img_lvl.transpose_(1,2).transpose_(0,1).contiguous()
        img_pyr.append(img_lvl)
    return img_pyr