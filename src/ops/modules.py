import torch.nn as nn
import torch
import numpy as np


class Pixel_Shuffle(nn.Module):
    def __init__(self, scale):
        super(Pixel_Shuffle, self).__init__()
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        scale2 = self.scale ** 2
        res = x.reshape(b, c // scale2, self.scale, self.scale, h, w).permute(0, 1, 4, 2, 5, 3) \
            .reshape(b, c // scale2, h * self.scale, w * self.scale)
        return res

# Referece:
# https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=1, kernel=3, dilation=1, bias=False, norm='none'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        # inp and oup is different, so use_res_connet is false
        self.use_res_connect = self.stride == 1 and inp == oup

        padding = ((kernel + ((dilation - 1) * (kernel - 1))) - 1) // 2
        norm_fn = get_norm_function(norm)
        # self.b1 = norm_fn(hidden_dim)
        # self.b2 = norm_fn(oup)
        if expand_ratio == 1:
            if norm !='none':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, dilation=dilation, groups=hidden_dim,
                            bias=bias),
                    norm_fn(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
                    norm_fn(oup)
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, dilation=dilation, groups=hidden_dim,
                            bias=bias),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
                )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=bias),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, dilation=dilation, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DecoderResBlock_depthwise(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, kernel=3, dilation=1, bias=False, use_res_connect=True, skip_relu=False, norm='none'):
        super(DecoderResBlock_depthwise, self).__init__()
        if skip_relu:
            self.layers = nn.Sequential(
                InvertedResidual(in_ch, in_ch * 2, 1, kernel=kernel, dilation=dilation, bias=bias, norm=norm),
                InvertedResidual(in_ch * 2, out_ch, 1, kernel=kernel, dilation=dilation, bias=bias, norm=norm))
        else:
            self.layers = nn.Sequential(
                InvertedResidual(in_ch, in_ch * 2, 1, kernel=kernel, dilation=dilation, bias=bias, norm=norm), nn.ReLU(),
                InvertedResidual(in_ch * 2, out_ch, 1, kernel=kernel, dilation=dilation, bias=bias, norm=norm))
        self.use_res_connect = use_res_connect

    def forward(self, x):
        x_input = x
        x = self.layers(x)
        if self.use_res_connect:
            x = x + x_input
        return x


class BasicBlockWoBN(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True, **kwargs):
        super(BasicBlockWoBN, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=True)
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, x):
        identity = x
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x += identity
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.b2(x)
        x += identity
        x = self.relu(x)
        return x


class ConvBNReLULayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBNReLULayer, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, scale=4, num_feat=64,
                 kernel=3, in_chn=3, out_chn=3, layer_num=[3, 2, 3], block_type='depthwise',
                 bias=False, use_res_connect2=True, use_res_connect1=True,
                 fix_layer=[False, False, False, False, False],
                 skip_outconv=False, skip_cat=False, skip_relu=False, clamp=True, norm='none', kaiming_normal=True):
        super(Decoder, self).__init__()

        self.in_chn = in_chn
        self.conv1 = nn.Conv2d(in_chn, num_feat, 3, padding=1)
        self.use_res_connect1 = use_res_connect1
        self.skip_outconv = skip_outconv
        self.skip_cat = skip_cat
        self.clamp = clamp
        # print("self.res_layer1 = self._make_layer(DecoderResBlock_depthwise, num_feat, num_feat, 3)")
        self.layer_num = layer_num
        if block_type == 'depthwise':
            basic_layer = DecoderResBlock_depthwise
        elif block_type == 'conv3x3':
            basic_layer = BasicBlockWoBN
        self.res_layer1 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[0], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=norm)
        self.res_layer2 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[1], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=norm)
        self.res_layer3 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[2], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=norm)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feat, out_chn * (scale ** 2), 3, padding=1, bias=bias)
        )

        self.d2s = Pixel_Shuffle(scale)
        if self.clamp:
            self.sigmoid = nn.Sigmoid()
        if kaiming_normal:
            self.reset_params()

        self.layer_list = [self.conv1, self.res_layer1, self.res_layer2, self.res_layer3, self.conv2]
        assert len(fix_layer) == len(self.layer_list)
        for i in np.arange(len(fix_layer)):
            if fix_layer[i]:
                for k, v in self.layer_list[i].named_parameters():
                    v.requires_grad = False

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def _make_layer(self, block, in_ch, out_ch, blocks_num, bias=False, use_res_connect=True, skip_relu=False,
                    kernel=3, norm='none'):
        layers = []
        for _ in range(blocks_num):
            layers.append(
                block(in_ch, out_ch, bias=bias, use_res_connect=use_res_connect, skip_relu=skip_relu, kernel=kernel, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x, clamp=True):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''

        x = self.conv1(x)

        if self.layer_num[0] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer1(x)
            if self.use_res_connect1: x += identity

        if self.layer_num[1] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer2(x)
            if self.use_res_connect1: x += identity

        if self.layer_num[2] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer3(x)
            if self.use_res_connect1: x += identity

        if self.skip_outconv:
            x = x[:, :48, :, :]
        else:
            x = self.conv2(x)

        if self.clamp:
            x = self.sigmoid(x)
        # TODO: remove the below line when clean the code
        if not self.skip_cat:
            x = torch.cat([x[:, i::self.in_chn, ...] for i in range(self.in_chn)], dim=1)
        x = self.d2s(x)

        return x

class Decoder_default_init_conv_between_res(nn.Module):
    def __init__(self, scale=4, num_feat=64,
                 kernel=3, in_chn=3, out_chn=3, layer_num=[3, 2, 3], block_type='depthwise',
                 bias=False, use_res_connect2=True, use_res_connect1=True,
                 fix_layer=[False, False, False, False, False],
                 skip_outconv=False, skip_cat=False, skip_relu=False, clamp=False, norm='none', kaiming_normal=False):
        super(Decoder_default_init_conv_between_res, self).__init__()

        self.in_chn = in_chn
        self.conv1 = nn.Conv2d(in_chn, num_feat, 3, padding=1)
        self.use_res_connect1 = use_res_connect1
        self.skip_outconv = skip_outconv
        self.skip_cat = skip_cat
        self.clamp = clamp
        # print("self.res_layer1 = self._make_layer(DecoderResBlock_depthwise, num_feat, num_feat, 3)")
        self.layer_num = layer_num
        if block_type == 'depthwise':
            basic_layer = DecoderResBlock_depthwise
        elif block_type == 'conv3x3':
            basic_layer = BasicBlockWoBN
        self.res_layer1 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[0], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=norm)
        self.res_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.res_layer2 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[1], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=norm)
        self.res_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.res_layer3 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[2], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=norm)
        self.res_conv3 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feat, out_chn * (scale ** 2), 3, padding=1, bias=bias)
        )

        self.d2s = Pixel_Shuffle(scale)
        if self.clamp:
            self.sigmoid = nn.Sigmoid()
        if kaiming_normal:
            self.reset_params()

        self.layer_list = [self.conv1, self.res_layer1, self.res_layer2, self.res_layer3, self.conv2]
        assert len(fix_layer) == len(self.layer_list)
        for i in np.arange(len(fix_layer)):
            if fix_layer[i]:
                for k, v in self.layer_list[i].named_parameters():
                    v.requires_grad = False

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def _make_layer(self, block, in_ch, out_ch, blocks_num, bias=False, use_res_connect=True, skip_relu=False,
                    kernel=3, norm='none'):
        layers = []
        for _ in range(blocks_num):
            layers.append(
                block(in_ch, out_ch, bias=bias, use_res_connect=use_res_connect, skip_relu=skip_relu, kernel=kernel, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x, clamp=True):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''

        x = self.conv1(x)

        if self.layer_num[0] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer1(x)
            x = self.res_conv1(x)
            if self.use_res_connect1: x += identity

        if self.layer_num[1] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer2(x)
            x = self.res_conv2(x)
            if self.use_res_connect1: x += identity

        if self.layer_num[2] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer3(x)
            x = self.res_conv3(x)
            if self.use_res_connect1: x += identity

        if self.skip_outconv:
            x = x[:, :48, :, :]
        else:
            x = self.conv2(x)

        if self.clamp:
            x = self.sigmoid(x)
        # TODO: remove the below line when clean the code
        if not self.skip_cat:
            x = torch.cat([x[:, i::self.in_chn, ...] for i in range(self.in_chn)], dim=1)
        x = self.d2s(x)

        return x


class Decoder_single_channel(nn.Module):

    def __init__(self, scale=4, num_feat=64,
                 kernel=3, in_chn=64, out_chn=64, layer_num=[3, 2, 3], block_type='depthwise',
                 bias=True, use_res_connect2=True, use_res_connect1=True,
                 skip_relu=False, norm='none'):
        """
        norm is in "bn", "in", "identity", "statistics", or [128, 1.0], or [2550.0, 10.0]


        """
        super(Decoder_single_channel, self).__init__()

        self.in_chn = in_chn
        self.conv1 = nn.Conv2d(in_chn, num_feat, kernel_size=1, padding=0, bias=bias)
        self.use_res_connect1 = use_res_connect1

        self.norm = norm
        self.norm_func_name = 'none'
        self.input_norm = None
        self.output_norm = None
        if isinstance(norm, str):
            assert norm in ["bn", "in", "identity", "none", "statistics"], f"norm type {norm} is not supported"
            if norm == 'statistics':
                var_path = 'Experimental_root/archs/channel_stdv_10percent_div2k.npy'
                # var = np.load(var_path)
                self.input_norm = torch.tensor(np.load(var_path).reshape(1, -1, 1, 1)).cuda()
                self.output_norm = 10.0
            else:
                self.norm_func_name = norm
                self.norm_input = get_norm_function(self.norm_func_name)(in_chn)
                self.norm_output = get_norm_function(self.norm_func_name)(out_chn)
        elif isinstance(norm, list):
            assert len(norm) == 2, "for input and output, the normalization has length 2 "
            self.input_norm = norm[0]
            self.output_norm = norm[1]            
        else:
            norm = [2550.0, 10.0]
            self.input_norm = norm[0]
            self.output_norm = norm[1]
            # raise ValueError('norm type {norm} is not supported')
        # self.norm_input = getnorm(in_chn)
        

        # self.norm_function = get_norm_function(norm)
        
        self.layer_num = layer_num
        if block_type == 'depthwise':
            basic_layer = DecoderResBlock_depthwise
        elif block_type == 'conv3x3':
            basic_layer = BasicBlockWoBN
        elif block_type == 'conv_bn_relu':
            basic_layer = ConvBNReLULayer

        self.res_layer1 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[0], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=self.norm_func_name)
        self.res_layer2 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[1], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=self.norm_func_name)
        self.res_layer3 = self._make_layer(basic_layer, num_feat, num_feat, layer_num[2], bias=bias,
                                           use_res_connect=use_res_connect2, skip_relu=skip_relu, kernel=kernel, norm=self.norm_func_name)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feat, out_chn * (scale ** 2), kernel_size=1, padding=0, bias=bias)
        )

        self.d2s = nn.PixelShuffle(scale)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def _make_layer(self, block, in_ch, out_ch, blocks_num, bias=False, use_res_connect=True, skip_relu=False,
                    kernel=3, norm='none'):
        layers = []
        for _ in range(blocks_num):
            layers.append(
                block(in_ch, out_ch, bias=bias, use_res_connect=use_res_connect, skip_relu=skip_relu, kernel=kernel, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''
        if self.input_norm is not None:
            if self.norm == 'statistics':
                x = x/self.input_norm/10.0
            else:
                x = x/self.input_norm
        if self.norm_func_name != 'none': # [4, 64, 8, 8]
            x = self.norm_input(x)

        x = self.conv1(x)
        
        if self.layer_num[0] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer1(x)
            if self.use_res_connect1: x += identity
            # x = identity + 0.01*x
        if self.layer_num[1] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer2(x)
            if self.use_res_connect1: x += identity

        if self.layer_num[2] != 0:
            if self.use_res_connect1: identity = x
            x = self.res_layer3(x)
            if self.use_res_connect1: x += identity

        x = self.conv2(x)
        x = self.d2s(x)

        if self.norm_func_name != 'none':
            x = self.norm_output(x)
        if self.output_norm is not None:
             x = x/self.output_norm

        return x


######################################## ENCODER #############################################
######################################## ENCODER #############################################
######################################## ENCODER #############################################


class SqueezeLayer(nn.Module):
    def __init__(self, factor, order):
        super().__init__()
        self.factor = factor
        self.order = order

    def forward(self, input, rev=False):
        if not rev:
            output = self.squeeze2d(input, self.factor, self.order)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input, self.factor, self.order)
            return output

    @staticmethod
    def squeeze2d(input, factor, order):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        B, C, H, W = input.shape
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)

        if order == 'ref':
            forw_order = [i * C // 3 + j for j in range(C // 3) for i in range(4)]
            x = x.reshape([B, C * 4 // 3, 3, H // 2, W // 2])
            x = x[:, forw_order, :, :, :]
            x = x.reshape([B, C * 4, H // 2, W // 2])
        elif order == 'hl':
            pass
        return x

    @staticmethod
    def unsqueeze2d(input, factor, order):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert C % (factor2) == 0, "{}".format(C)

        if order == 'ref':
            back_order = [i + j * 4 for i in range(4) for j in range(C // factor2 // 3)]
            x = input.reshape([B, C // 3, 3, H, W])
            x = x[:, back_order, :, :, :]
            x = x.reshape([B, C, H, W])
        elif order == 'hl':
            x = input
        x = x.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x


def get_norm_function(norm):
    if norm == "bn":
        norm_fn = nn.BatchNorm2d
    elif norm == "in":
        norm_fn = nn.InstanceNorm2d
    elif norm == 'none':
        norm_fn = nn.Identity
    elif norm == 'identity':
        norm_fn = nn.Identity
    else:
        norm_fn = norm 
    return norm_fn


def get_act_function(act):
    if act == "ReLU":
        act_fn = nn.ReLU
    elif act == "ReLU6":
        act_fn = nn.ReLU6
    else:
        act_fn = nn.Identity
    return act_fn


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True, act='LeakyReLU'):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = get_act_function(act)()
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def _make_layer_act(block, in_ch, out_ch, blocks_num, act='LeakyReLU'):
    layers = []
    for _ in range(blocks_num):
        layers.append(block(in_ch, out_ch, out_ch, act=act))
    return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm='bn', act='ReLU'):
        super(DoubleConv, self).__init__()
        norm_fn = get_norm_function(norm)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, stride=stride, bias=False)
        self.b1 = norm_fn(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.b2 = norm_fn(out_ch)
        self.relu = get_act_function(act)(inplace=False)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu(x)
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
    def __init__(self, num_feat, dense_block_num, norm='bn', act='ReLU') -> None:
        super().__init__()
        self.down_double_conv1 = DoubleConv(num_feat, num_feat * 2, stride=2, norm=norm, act=act)
        self.res_layer4 = _make_layer_act(DenseBlock, num_feat * 2, num_feat * 2, dense_block_num, act=act)

    def forward(self, x):
        x1 = self.down_double_conv1(x)
        x1 = self.res_layer4(x1)
        return x1


class UpsampleDenseBlock(nn.Module):
    def __init__(self, num_feat, dense_block_num, norm='bn', act='ReLU') -> None:
        super().__init__()
        self.up1 = UpBlock(num_feat * 2, num_feat)
        self.double_conv1 = DoubleConv(num_feat * 2, num_feat, norm=norm, act=act)
        self.res_layer5 = _make_layer_act(DenseBlock, num_feat, num_feat, dense_block_num, act=act)

    def forward(self, x1, x2):
        x3 = self.double_conv1(torch.cat([x1, self.up1(x2)], dim=1))
        x3 = self.res_layer5(x3)
        return x3


class DenseUnetDeepEncoder(nn.Module):
    def __init__(self, scale=4, num_feat=64, out_chn=3, input_chn=3,
                 dense_block_num=1, down_layer=2, norm='bn', act='ReLU'):
        super(DenseUnetDeepEncoder, self).__init__()

        self.scale = scale
        self.num_feat = num_feat
        self.down_layer = down_layer

        # space to depth
        self.s2d = SqueezeLayer(scale, 'hl')

        # start residual blocks: conv -> relu -> conv
        self.conv_first = nn.Conv2d(input_chn * scale ** 2, num_feat, 3, padding=1)
        self.relu = get_act_function(act)(inplace=True)
        self.res_layer1 = _make_layer_act(DenseBlock, num_feat, num_feat, dense_block_num, act=act)

        # Downsample
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        for i in np.arange(down_layer):
            self.down_list.append(DownsampleDenseBlock(num_feat * (2 ** i), dense_block_num, norm=norm, act=act))
            self.up_list.append(UpsampleDenseBlock(num_feat * (2 ** i), dense_block_num, norm=norm, act=act))

        # Upsample
        self.conv_last = nn.Conv2d(num_feat, out_chn, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''
        x = self.s2d(x)

        x = self.relu(self.conv_first(x))

        # start residual blocks
        x = self.res_layer1(x)
        feature_list = [x]
        # U-Net
        # down size
        for i in np.arange(self.down_layer):
            feature_list.append(x)
            x = self.down_list[i](x)

        # up size
        for i in np.arange(self.down_layer):
            x = self.up_list[self.down_layer - i - 1](feature_list.pop(), x)

        x = self.sigmoid(self.conv_last(x))
        # TODO: sigmoid -> torch.clamp() 可能会导致性能问题

        return x

########################################## compression related modules #############################################

