# Standard libraries
import numpy as np
import copy
# PyTorch
import torch
import torch.nn as nn

JPEG_Y_TABLE = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

#
JPEG_C_TABLE = np.ones((8, 8), dtype=np.float32) * 99
JPEG_C_TABLE[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                 [24, 26, 56, 99], [47, 66, 99, 99]]).T

def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


def write_jpeg(quant_tables, dct_coeffs, path):
    # TODO
    raise NotImplementedError


def read_jpeg(path):
    # TODO
    raise NotImplementedError


def skip_chroma_sub(x):
    return x

def default_chroma_sub(cbcr):
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                            count_include_pad=False)
    cbcr = avg_pool(cbcr)
    return cbcr


def default_chroma_up(cbcr):
    b, c, h, w = cbcr.shape
    cbcr = cbcr.view(b, c, h, 1, w, 1)
    # repeat
    cbcr = cbcr.repeat(1, 1, 1, 2, 1, 2).view(b, c, h * 2, w * 2)
    return cbcr


def image2block(x, block_size=8):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // block_size, block_size, w // block_size, block_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, block_size, block_size) # b*nH*nW, c, 8, 8
    return x


def block2image(x, shape, block_size=8):
    b, c, h, w = shape
    x = x.reshape(b, h // block_size, w // block_size, c, block_size, block_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(shape)
    return x


class color_cvt_util(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(color_cvt_util, self).__init__()
        matrix_rgb2ycbcr = np.array(
            [[0.299, 0.587, 0.114],
             [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T

        self.matrix_rgb2ycbcr = torch.tensor(matrix_rgb2ycbcr, requires_grad=False)

        matrix_ycbcr2rgb = np.array(
            [[1., 0., 1.402],
             [1, -0.344136, -0.714136],
             [1, 1.772, 0]], dtype=np.float32).T

        self.matrix_ycbcr2rgb = torch.tensor(matrix_ycbcr2rgb, requires_grad=False)
        self.shift = torch.tensor([0., 128., 128.])

    def rgb2ycbcr(self, image):
        image = image.permute(0, 2, 3, 1)  # B, H, W, C
        result = torch.tensordot(image, self.matrix_rgb2ycbcr.to(image.device), dims=1) + self.shift.to(image.device)
        result = result.permute(0, 3, 1, 2)  # B C H W
        return result

    def ycbcr2rgb(self, image):
        image = image.permute(0, 2, 3, 1)  # B, H, W, C
        result = torch.tensordot(image - self.shift.to(image.device), self.matrix_ycbcr2rgb.to(image.device), dims=1)
        result = result.permute(0, 3, 1, 2)
        return result
