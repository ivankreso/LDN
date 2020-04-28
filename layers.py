import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

checkpoint = lambda func, *inputs: cp.checkpoint(func, *inputs, preserve_rng_state=False)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, conv_class, upsample_func, num_maps_in, bt_size=512, level_size=128,
                 out_size=256, grids=[6,3,2,1], square_grid=False):
        super(SpatialPyramidPooling, self).__init__()
        self.upsample = upsample_func
        self.grids = grids
        self.num_levels = len(grids)
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', conv_class(num_maps_in, bt_size, k=1))
        num_features = bt_size
        final_size = num_features
        for i in range(self.num_levels):
            final_size += level_size
            self.spp.add_module('spp'+str(i), conv_class(num_features, level_size, k=1))
        self.spp.add_module('spp_fuse', conv_class(final_size, out_size, k=1))


    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i-1], max(1, round(ar*self.grids[i-1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i-1])
            level = self.spp[i].forward(x_pooled)
            level = self.upsample(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        return self.spp[-1].forward(x)



class Upsample(nn.Module):
    def __init__(self, conv_class, upsample_func, num_maps_in, skip_maps_in, num_maps_out, k,
                 produce_aux=False, num_classes=0, dws_conv=False, checkpointing=False):
        super(Upsample, self).__init__()
        print('Upsample layer: in =', num_maps_in, ', skip =', skip_maps_in, ' out =', num_maps_out)
        self.upsample_func = upsample_func
        self.bottleneck = conv_class(skip_maps_in, num_maps_in, k=1)
        self.produce_aux = produce_aux
        self.has_blend_conv = num_maps_out > 0
        self.num_maps_out = num_maps_in
        self.checkpointing = checkpointing
        if produce_aux:
            self.aux_logits = conv_class(num_maps_in, num_classes, k=1, output_conv=True)
        if self.has_blend_conv:
            self.num_maps_out = num_maps_out
            bt_maps = 128
            self.blend_bt = None
            if not dws_conv and k >=3 and num_maps_in > bt_maps:
                print(f'Bottleneck before 3x3: {num_maps_in} -> {bt_maps}')
                self.blend_bt = conv_class(num_maps_in, bt_maps, k=1)
                num_maps_in = bt_maps
            self.blend_conv = conv_class(num_maps_in, num_maps_out, k=k)
        self.forward_func = self._get_forward_func()

    def _get_forward_func(self):
        def func(*inputs):
            x, skip = inputs
            skip = self.bottleneck(skip)
            skip_size = skip.size()[2:4]
            if self.produce_aux:
                aux = self.aux_logits(x)
            x = self.upsample_func(x, skip_size)
            x += skip
            if self.has_blend_conv:
                if self.blend_bt is not None:
                    x = self.blend_bt(x)
                x = self.blend_conv(x)
            if self.produce_aux:
                return x, aux
            return x
        return func

    def forward(self, bottom, skip):
        if self.checkpointing and self.training:
            return checkpoint(self.forward_func, *[bottom, skip])
        else:
            return self.forward_func(*[bottom, skip])


class UpsampleResidual(nn.Module):
    def __init__(self, conv_class, upsample_func, num_maps_in, skip_maps_in, num_maps_out, k,
                 produce_aux=False, num_classes=0, dws_conv=False):
        super(UpsampleResidual, self).__init__()
        print('Upsample layer: in =', num_maps_in, ', skip =', skip_maps_in, ' out =', num_maps_out)
        self.upsample_func = upsample_func
        self.bottleneck = conv_class(skip_maps_in, num_maps_in, k=1)
        self.produce_aux = produce_aux
        self.has_blend_conv = num_maps_out > 0
        self.num_maps_out = num_maps_in
        if num_maps_out != num_maps_in:
            self.skip_bt = conv_class(num_maps_in, num_maps_out, k=1)
            print(f'Bottleneck on residual: {num_maps_in} -> {num_maps_out}')
        else:
            self.skip_bt = None

        if produce_aux:
            self.aux_logits = conv_class(num_maps_in, num_classes, k=1, output_conv=True)

        if self.has_blend_conv:
            self.num_maps_out = num_maps_out
            bt_maps = 128
            self.blend_bt = None
            if not dws_conv and k >=3 and num_maps_in > bt_maps:
                print(f'Bottleneck before 3x3: {num_maps_in} -> {bt_maps}')
                self.blend_bt = conv_class(num_maps_in, bt_maps, k=1)
                num_maps_in = bt_maps
            self.blend_conv = conv_class(num_maps_in, num_maps_out, k=k)


    def forward(self, bottom, skip):
        skip = self.bottleneck(skip)
        skip_size = skip.size()[2:4]
        if self.produce_aux:
            aux = self.aux_logits(bottom)

        bottom = self.upsample_func(bottom, skip_size)
        x = skip
        x += bottom

        if self.has_blend_conv:
            if self.blend_bt is not None:
                x = self.blend_bt(x)
            x = self.blend_conv(x)
        if self.skip_bt is not None:
            bottom = self.skip_bt(bottom)
        x += bottom
        if self.produce_aux:
            return x, aux
        return x