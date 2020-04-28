import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import torch.utils.checkpoint as cp

import numpy as np
import PIL.Image as pimg

import layers
import losses
import utils
import evaluation
import data.data_utils as data_utils
import data.transform as transform

evaluate = evaluation.evaluate_semseg

use_pyl_in_spp = True
checkpoint_stem = False
checkpoint_upsample = False
use_dws_up = False
use_dws_down = False

first_stride = 2
avg_pooling_k = 2
upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)

checkpoint = lambda func, *inputs: cp.checkpoint(func, *inputs, preserve_rng_state=True)

def _batchnorm_factory(num_maps, momentum):
    return nn.BatchNorm2d(num_maps, eps=1e-5, momentum=momentum)


use_batchnorm = True
drop_rate = 0.0

def build(depth, pretrained, **kwargs):
    global imagenet_init
    imagenet_init = pretrained
    if depth == 32:
        return densenet32(**kwargs)
    if depth == 60:
        return densenet60(**kwargs)
    if depth == 121:
        return densenet121(**kwargs)
    elif depth == 169:
        return densenet169(**kwargs)
    elif depth == 201:
        return densenet201(**kwargs)
    elif depth == 161:
        return densenet161(**kwargs)
    else:
        raise ValueError('Unknown model variant')

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, **kwargs)
    return model

def _load_imagenet_weights(name):
    params = model_zoo.load_url(model_urls[name])
    delete_keys = ['features.norm5.weight', 'features.norm5.bias',
    'features.norm5.running_mean', 'features.norm5.running_var',
    'classifier.weight', 'classifier.bias']
    for key in delete_keys:
        del params[key]
    return params


def densenet_small(**kwargs):
    assert imagenet_init == 0
    model = DenseNet(num_init_features=64, growth_rate=64, bn_size=2,
                     block_config=(3, 6, 12, 8), **kwargs)
    return model


def densenet32(**kwargs):
    assert imagenet_init == 0
    model = DenseNet(num_init_features=64, growth_rate=64, bn_size=2,
                     block_config=(2, 4, 6, 4), **kwargs)
    return model


def densenet60(**kwargs):
    assert imagenet_init == 0
    model = DenseNet(num_init_features=64, growth_rate=64, bn_size=2,
                     block_config=(3, 6, 12, 8), **kwargs)
    return model


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
            imagenet_init (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_init:
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 24, 16), **kwargs)
        params = _load_imagenet_weights('densenet121')
        model.load_state_dict(params)
    else:
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 24, 16), drop_rate=drop_rate, **kwargs)
    return model


def densenet169(**kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
            imagenet_init (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_init:
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 32, 32), **kwargs)
        params = _load_imagenet_weights('densenet169')
        model.load_state_dict(params)
    else:
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 32, 32), drop_rate=0.2, **kwargs)
    return model


def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
            imagenet_init (bool): If True, returns a model pre-trained on ImageNet
    """
    assert imagenet_init
    if imagenet_init:
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 48, 32), **kwargs)
        params = _load_imagenet_weights('densenet201')
        model.load_state_dict(params)
    return model


def densenet161(**kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
            imagenet_init (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_init:
        model = DenseNet(num_init_features=96, growth_rate=48,
                         block_config=(6, 12, 36, 24), **kwargs)
        params = _load_imagenet_weights('densenet161')
        model.load_state_dict(params)
    else:
        model = DenseNet(num_init_features=96, growth_rate=48,
                         block_config=(6, 12, 36, 24), drop_rate=0.2, **kwargs)

    return model


def _checkpoint_unit_nobt(bn, relu, conv):
    def func(*x):
        x = torch.cat(x, 1)
        return conv(relu(bn(x)))
    return func

def _checkpoint_unit(bn1, relu1, conv1, bn2, relu2, conv2):
    def func(*x):
        x = torch.cat(x, 1)
        x = conv1(relu1(bn1(x)))
        return conv2(relu2(bn2(x)))
    return func

def _checkpoint_block_func(block):
    def func(x):
        return block(x)
    return func


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        assert kernel_size > 1
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                              padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self,x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 dilation=1, checkpointing=False):
        super(_DenseLayer, self).__init__()
        m = batchnorm_momentum_ckpt if checkpointing else batchnorm_momentum

        if not use_dws_down:
            bottleneck_size = bn_size * growth_rate
            if imagenet_init or num_input_features > bottleneck_size:
                if use_batchnorm:
                    self.add_module('norm1', _batchnorm_factory(num_input_features, m))
                self.add_module('relu1', nn.ReLU(inplace=True)),
                self.add_module('conv1', nn.Conv2d(num_input_features, bottleneck_size,
                                kernel_size=1, stride=1, bias=False))
                num_feats = bottleneck_size
            else:
                num_feats = num_input_features
            if use_batchnorm:
                self.add_module('norm2', _batchnorm_factory(num_feats, m))
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(num_feats, growth_rate, kernel_size=3,
                            stride=1, padding=dilation, bias=False, dilation=dilation))
        else:
            num_feats = num_input_features
            if use_batchnorm:
                self.add_module('norm2', _batchnorm_factory(num_feats, m))
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', SeparableConv2d(num_feats, growth_rate, kernel_size=3,
                            stride=1, padding=dilation, bias=False, dilation=dilation))

        self.drop_rate = drop_rate
        self.checkpointing = checkpointing
        if checkpointing:
            if len(self) == 6:
                self.conv_func = _checkpoint_unit(self.norm1, self.relu1, self.conv1,
                                                  self.norm2, self.relu2, self.conv2)
            else:
                self.conv_func = _checkpoint_unit_nobt(self.norm2, self.relu2, self.conv2)

    def forward(self, x):
        if self.checkpointing:
            if self.training:
                x = checkpoint(self.conv_func, *x)
            else:
                x = self.conv_func(*x)
        else:
            x = super(_DenseLayer, self).forward(x)

        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training, inplace=True)

        return x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,
                 split=False, dilation=1, checkpointing=True):
        super(_DenseBlock, self).__init__()
        self.checkpointing = checkpointing
        self.split = split

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size,
                drop_rate=drop_rate, dilation=dilation, checkpointing=checkpointing)
            self.add_module('denselayer%d' % (i + 1), layer)
        if split:
            self.split_size = num_input_features + (num_layers // 2) * growth_rate
            k = avg_pooling_k
            pad = (k-1) // 2
            self.pool_func = lambda x: F.avg_pool2d(x, k, 2, padding=pad, ceil_mode=False,
                                                    count_include_pad=False)

    def forward(self, x):
        if self.checkpointing:
            x = [x]
        for i, layer in enumerate(self.children()):
            if self.split and len(self) // 2 == i:
                if self.checkpointing:
                    split = torch.cat(x, 1)
                    x = [self.pool_func(split)]
                else:
                    split = x
                    x = self.pool_func(split)
            if self.checkpointing:
                x.append(layer(x))
            else:
                x = torch.cat([x, layer(x)], 1)
        if self.checkpointing:
            x = torch.cat(x, 1)
        if self.split:
            return x, split
        else:
            return x


class _Transition(nn.Sequential):
    @staticmethod
    def _checkpoint_function(bn, relu, conv, pool):
        def func(inputs):
            return pool(conv(relu(bn(inputs))))
        return func

    def __init__(self, num_input_features, num_output_features, stride=2, checkpointing=False):
        super(_Transition, self).__init__()
        self.stride = stride
        if use_batchnorm:
            m = batchnorm_momentum_ckpt if checkpointing else batchnorm_momentum
            self.add_module('norm', _batchnorm_factory(num_input_features, m))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if stride > 1:
            if avg_pooling_k == 2:
                self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))
            elif avg_pooling_k == 3:
                self.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=stride,
            					padding=1, ceil_mode=False, count_include_pad=False))
        else:
            self.pool = lambda x: x
        self.checkpointing = checkpointing
        if checkpointing:
            self.func = _Transition._checkpoint_function(self.norm, self.relu, self.conv, self.pool)

    def forward(self, x):
        if self.checkpointing and self.training:
            return checkpoint(self.func, x)
        else:
            return super(_Transition, self).forward(x)


class _BNReluConv(nn.Sequential):
    @staticmethod
    def _checkpoint_function(bn, relu, conv):
        def func(inputs):
            return conv(relu(bn(inputs)))
        return func

    def __init__(self, num_maps_in, num_maps_out, k=3, output_conv=False,
                 dilation=1, drop_rate=0, checkpointing=False):
        super(_BNReluConv, self).__init__()
        self.drop_rate = drop_rate
        if use_batchnorm:
            m = batchnorm_momentum_ckpt if checkpointing else batchnorm_momentum
            self.add_module('norm', _batchnorm_factory(num_maps_in, m))
        self.add_module('relu', nn.ReLU(inplace=True))
        padding = ((k-1) // 2) * dilation
        if k >= 3 and use_dws_up:
            self.add_module('conv', SeparableConv2d(num_maps_in, num_maps_out, kernel_size=k,
                padding=padding, bias=output_conv, dilation=dilation))
        else:
            self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k,
                            padding=padding, bias=output_conv, dilation=dilation))
        self.checkpointing = checkpointing
        if checkpointing:
            self.func = _BNReluConv._checkpoint_function(self.norm, self.relu, self.conv)

    def forward(self, x):
        if self.checkpointing and self.training:
            x = checkpoint(self.func, x)
        else:
            x = super(_BNReluConv, self).forward(x)
        return x


class SequentialPrint(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
            print(module)
            print(input.size())
        return input


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
                (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
    """
    def __init__(self, dataset, args, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()
        global batchnorm_momentum, batchnorm_momentum_ckpt
        batchnorm_momentum = dataset.batchnorm_momentum
        print('BatchNorm momentum =', batchnorm_momentum)
        self.checkpointing = args.checkpointing
        self.checkpoint_stem = checkpoint_stem
        if self.checkpointing or self.checkpoint_stem:
            batchnorm_momentum_ckpt = min(np.roots([-1, 2, -batchnorm_momentum]))
            print('BatchNorm momentum ckpt =', batchnorm_momentum_ckpt)
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.num_logits = dataset.num_logits
        self.args = args
        self.block_config = block_config
        self.num_blocks = len(block_config)
        self.growth_rate = growth_rate

        # First convolution
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=7,
                                 stride=first_stride, padding=3, bias=False))
        if imagenet_init and use_batchnorm:
            m = batchnorm_momentum_ckpt if self.checkpoint_stem else batchnorm_momentum
            self.features.add_module('norm0', _batchnorm_factory(num_init_features, m))
            self.features.add_module('relu0', nn.ReLU(inplace=True))

        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2))
        self.first_block_idx = len(self.features)
        if self.checkpoint_stem:
            self.first_ckpt_func = self._checkpoint_segment(0, self.first_block_idx)

        self.random_init = []
        self.fine_tune = []
        if args.pretrained:
            self.fine_tune.append(self.features)
        else:
            self.random_init.append(self.features)

        splits = [False, False, True, False]
        up_sizes = [256, 256, 128, 128]

        # for cityscapes/vistas
        spp_square_grid = False
        spp_grids = [8,4,2,1]
        # spp_grids = [4,2,1]

        # for VOC2012
        # spp_square_grid = True
        # spp_grids = [6,3,2,1]

        self.spp_size = 512
        bt_size = 512
        level_size = 128
        dilations = [1] * len(block_config)
        strides = [2] * (len(block_config) - 1)

        # dilate last block
        # dilations = [1, 1, 1, 2]
        # strides = [2, 2, 1]
        # up_sizes = up_sizes[1:]

        # dilate last two blocks
        # dilations = [1, 1, 2, 4]
        # strides = [2, 1, 1]
        # up_sizes = up_sizes[2:]

        num_downs = first_stride + strides.count(2) + sum(splits)
        num_ups = len(up_sizes)
        self.downsampling_factor = 2**num_downs
        self.upsampling_factor = 2**(num_downs-num_ups)
        print(self.downsampling_factor)
        print(self.upsampling_factor)
        args.downsampling_factor = self.downsampling_factor
        args.upsampling_factor = self.upsampling_factor

        self.use_upsampling_path = True

        skip_sizes = []
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers, num_input_features=num_features, bn_size=bn_size,
                growth_rate=growth_rate, drop_rate=drop_rate, split=splits[i],
                dilation=dilations[i], checkpointing=self.checkpointing)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if block.split and self.use_upsampling_path:
                skip_sizes.append(block.split_size)
            if i != len(block_config) - 1:
                if strides[i] > 1 and self.use_upsampling_path:
                    skip_sizes.append(num_features)
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2,
                    stride=strides[i], checkpointing=self.checkpointing)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.use_aux = args.aux_loss
        self.spp = layers.SpatialPyramidPooling(
            _BNReluConv, upsample, num_features, bt_size, level_size,
            self.spp_size, spp_grids, spp_square_grid)

        self.random_init.append(self.spp)
        num_features = self.spp_size

        if self.use_aux:
            self.pyramid_loss_scales = data_utils.get_pyramid_loss_scales(
                    args.downsampling_factor, args.upsampling_factor)
            if use_pyl_in_spp:
                spp_scales = []
                for scale in reversed(spp_grids):
                    assert args.crop_size % scale == 0
                    spp_scales.append(args.crop_size // scale)
                self.pyramid_loss_scales = spp_scales + self.pyramid_loss_scales

        if self.use_upsampling_path:
            up_convs = [3] * len(up_sizes)
            self.upsample_layers = nn.Sequential()
            self.random_init.append(self.upsample_layers)
            print(up_sizes, skip_sizes)
            assert len(up_sizes) == len(skip_sizes)
            for i in range(num_ups):
                upsample_unit = layers.UpsampleResidual(
                    _BNReluConv, upsample, num_features, skip_sizes[-1-i], up_sizes[i],
                    up_convs[i], self.args.aux_loss, self.num_classes)
                num_features = upsample_unit.num_maps_out
                self.upsample_layers.add_module('upsample_'+str(i), upsample_unit)
            print()

        self.logits = _BNReluConv(num_features, self.num_logits, k=1, output_conv=True,
                                  checkpointing=self.checkpointing)
        self.random_init.append(self.logits)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.logits.conv.weight.data)
        if self.use_aux and self.use_upsampling_path:
            for module in self.upsample_layers:
                nn.init.xavier_normal_(module.aux_logits.conv.weight.data)


    def _checkpoint_segment(self, start, end):
        def func(x):
            for i in range(start, end):
                x = self.features[i](x)
            return x
        return func


    def forward(self, x, target_size=None):
        skip_layers = []
        if target_size is None:
            target_size = x.size()[2:4]

        if not self.training or not self.checkpoint_stem:
            for i in range(self.first_block_idx+1):
                x = self.features[i].forward(x)
        else:
            x = checkpoint(self.first_ckpt_func, x)
            x = self.features[self.first_block_idx].forward(x)
        for i in range(self.first_block_idx+1, len(self.features), 2):
            if self.features[i].stride > 1 and self.use_upsampling_path:
                skip_layers.append(x)
            x = self.features[i].forward(x)
            x = self.features[i+1].forward(x)
            if isinstance(x, tuple) and self.use_upsampling_path:
                x, split = x
                skip_layers.append(split)

        x = self.spp(x)

        aux_logits = []
        if self.use_upsampling_path:
            for i, up in enumerate(self.upsample_layers):
                x = up(x, skip_layers[-1-i])
                if self.use_aux:
                    x, aux = x
                    aux_logits.append(aux)

        x = self.logits(x)
        x = upsample(x, target_size)

        return x, aux_logits


    def send_to_gpu(self, batch):
        x = batch['image'].cuda(non_blocking=True)
        batch['image'] = x
        batch['labels_cpu'] = batch['labels']
        batch['labels'] = batch['labels'].cuda(non_blocking=True)
        if self.use_aux and self.training:
            transform.build_pyramid_labels_th(batch, self.pyramid_loss_scales, self.num_classes)


    def forward_loss(self, batch, return_outputs=False):
        x = batch['image']
        logits, aux_logits = self.forward(x)
        if not self.training:
            aux_logits = []
        self.output = logits
        loss = losses.segmentation_loss(logits, aux_logits, batch, self.args.aux_loss_weight,
                                        self.dataset.ignore_id, equal_level_weights=use_pyl_in_spp)
        loss, self.aux_losses = loss
        if return_outputs:
            return loss, (logits, aux_logits)
        return loss


    def load_state_dict(self, state_dict, strict=False):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and persistent buffers.
        """
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print('Variable "{}" missing in model'.format(name))
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                      name, own_state[name].size(), param.size()))
                raise
        # missing = set(own_state.keys()) - set(state_dict.keys())
        # if len(missing) > 0:
        #         print('missing keys in state_dict: "{}"'.format(missing))
