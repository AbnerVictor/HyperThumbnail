import torch
from torch import nn as nn
import numpy as np
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Coeff_EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 in_range=1.0,
                 out_range=1.0,
                 mean=None):
        super(Coeff_EDSR, self).__init__()

        self.in_range = torch.tensor(in_range)
        self.out_range = torch.tensor(out_range)
        self.input_mean = torch.zeros(1, num_in_ch, 1, 1)
        self.output_mean = torch.zeros(1, num_out_ch, 1, 1)
        if isinstance(mean, tuple):
            self.mean = torch.Tensor(mean)
            self.input_mean = self.mean
            self.output_mean = self.mean
        elif mean == 'statistics':
            var_path = 'Experimental_root/archs/channel_stdv_10percent_div2k.npy'
            self.in_range = 1.0/torch.tensor(np.load(var_path).reshape(1, -1, 1, 1)).cuda()
            
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if upscale > 1:
            self.upsample = Upsample(upscale, num_feat)
        else:
            self.upsample = nn.Identity()
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.input_mean = self.input_mean.type_as(x)
        self.output_mean = self.output_mean.type_as(x)
        # x.var(axis=[0,2,3])
        x = (x - self.input_mean) * self.in_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.out_range + self.output_mean

        return x