#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.rrdbnet_arch import ResidualDenseBlock

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DoubleConv, self).__init__()
        # norm_fn = get_norm_function(norm)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, stride=stride, bias=False)
        
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.lrelu(x)
        x = self.c2(x)
        x = self.lrelu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class DownsampleDenseBlock(nn.Module):
    def __init__(self, num_feat, out_feat=None) -> None:
        super().__init__()
        if out_feat is None:
            out_feat = num_feat * 2
        self.down_double_conv1 = DoubleConv(num_feat, out_feat, stride=2)
        self.rdb = ResidualDenseBlock(num_feat=out_feat, num_grow_ch=num_feat)

    def forward(self, x):
        x1 = self.down_double_conv1(x)
        x1 = self.rdb(x1)
        return x1

class UpsampleDenseBlock(nn.Module):
    def __init__(self, num_feat) -> None:
        super().__init__()
        self.up1 = UpBlock(num_feat * 2, num_feat)
        self.double_conv1 = DoubleConv(num_feat * 2, num_feat)
        self.rdb = ResidualDenseBlock(num_feat=num_feat, num_grow_ch=num_feat)

    def forward(self, x1, x2):
        x3 = self.double_conv1(torch.cat([x1, self.up1(x2)], dim=1))
        x3 = self.rdb(x3)
        return x3
        
class RDBUnet(nn.Module):
    def __init__(self, scale=4, num_feat=64, out_chn=3, 
                input_chn=3, down_layer=2,
                rgb_mean=(0.4488, 0.4371, 0.4040),
                down_by_dense=False,
                slim_channel=False):
        super(RDBUnet, self).__init__()

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.scale = scale
        self.num_feat = num_feat
        self.down_layer = down_layer
        self.down_by_dense = down_by_dense
        # space to depth
        if self.down_by_dense:
            self.s2d = nn.ModuleList()
            if slim_channel:
                self.conv_first = nn.Conv2d(input_chn, num_feat//scale, 3, padding=1)
            else:
                self.conv_first = nn.Conv2d(input_chn, num_feat, 3, padding=1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            for i in np.arange(np.log2(scale)):
                if slim_channel:
                    self.s2d.append(DownsampleDenseBlock(int(num_feat *(2**i)//scale)))
                else:
                    self.s2d.append(DownsampleDenseBlock(num_feat, out_feat=num_feat))
        else:
            self.s2d = nn.PixelUnshuffle(scale)
            # start residual blocks: conv -> relu -> conv
            self.conv_first = nn.Conv2d(input_chn * scale ** 2, num_feat, 3, padding=1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.rdb_first = ResidualDenseBlock(num_feat=num_feat, num_grow_ch=num_feat//2)

        # Downsample
        self.down_list = nn.ModuleList()
        self.down_rdb_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.up_rdb_list = nn.ModuleList()
        for i in np.arange(down_layer):
            input_feat = num_feat * (2 ** i)
            self.down_list.append(DownsampleDenseBlock(input_feat))
            self.up_list.append(UpsampleDenseBlock(input_feat))

        # Upsample
        self.conv_last = nn.Conv2d(num_feat, out_chn, 3, padding=1, bias=False)


    def forward(self, x):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''
        self.mean = self.mean.type_as(x)
        x = x - self.mean
        

        if self.down_by_dense:
            x = self.lrelu(self.conv_first(x))
            for i in np.arange(int(np.log2(self.scale))):
                x = self.s2d[i](x)
        else:
            x = self.s2d(x)
            x = self.lrelu(self.conv_first(x))
            # start residual blocks
            x = self.rdb_first(x)
        
        feature_list = [x]
        # U-Net
        # down size
        for i in np.arange(self.down_layer):
            feature_list.append(x)
            x = self.down_list[i](x)

        # up size
        for i in np.arange(self.down_layer):
            x = self.up_list[self.down_layer - i - 1](feature_list.pop(), x)

        
        x = self.conv_last(x)
        x = x + self.mean

        return x

#%%
if __name__ == '__main__':
    model = RDBUnet()

    fake_input = torch.randn((1, 3, 256, 256)).float()
    out = model(fake_input)
    
    # print(jpeg['tables'])
    def print_stat(a): print(f"shape={a.shape}, min={a.min():.2f}, median={a.median():.2f}, max={a.max():.2f}, var={a.var():.2f}, {a.flatten()[0]}")
    print_stat(fake_input)
    print_stat(out)

# %%
