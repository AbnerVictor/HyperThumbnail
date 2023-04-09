import os
import torch.nn
from src.ops.JPEG_Utils.utils import *
from src.ops.JPEG_Utils.differentiable_quantize import choose_rounding
from src.ops.JPEG_Utils.dot_utils import dct, inv_dct, dct_matrix, zig_zag_flatten, zig_zag_unflatten
from basicsr.utils import get_root_logger

class DiffJPEG(nn.Module):
    def __init__(self, quant_type='gradient_1', optimize_table=True, init_table=None):
        """
        Args:
            quant_type: which quantization method to use
            optimize_table: require grads on quant table if True
            init_table: [Y_TABLE, C_TABLE], [2, 8, 8], init the quantization table with JPEG table by default
        """
        super(DiffJPEG, self).__init__()
        self.rounding = choose_rounding(quant_type)
        if init_table is None:
            self.Y_TABLE, self.C_TABLE = copy.deepcopy(JPEG_Y_TABLE), copy.deepcopy(JPEG_C_TABLE)
        else:
            self.Y_TABLE, self.C_TABLE = copy.deepcopy(init_table[0]), copy.deepcopy(init_table[1])

        self.Y_TABLE = nn.Parameter(self.rounding(torch.tensor(self.Y_TABLE)), requires_grad=not optimize_table)
        self.C_TABLE = nn.Parameter(self.rounding(torch.tensor(self.C_TABLE)), requires_grad=not optimize_table)

        self.dct_matrix = dct_matrix(8)
        self.dct_matrix.requires_grad = False

        self.color_cvt = color_cvt_util()
        self.color_cvt.requires_grad_(False)

    def dct(self, x):
        b, c, h, w = x.shape
        nH, nW = h // 8, w // 8
        # image 2 block
        x_blocks = image2block(x, block_size=8)

        x_dct = dct(x_blocks, self.dct_matrix.to(x.device))  # b*nH*nW，c, 8, 8
        x_dct = x_dct.reshape(b, nH, nW, c, 64).permute(0, 3, 4, 1, 2)  # b c 64 nH nW
        return x_dct

    def idct(self, x):
        b, c, _, nH, nW = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(b*nH*nW, c, 8, 8)
        x_idct = inv_dct(x, self.dct_matrix.to(x.device))  # b*nH*nW，c, 8, 8
        # block 2 image
        x_idct = block2image(x_idct, (b, c, nH*8, nW*8), block_size=8)

        return x_idct.reshape(b, c, nH * 8, nW * 8)

    def compress(self, x, quant_tables=None, chroma_sub=default_chroma_sub,
                 scale_chroma_sub=2, quality=50, range=255., skip_table_rounding=False):
        """

        Args:
            x: input image in [B, C, H, W], RGB color space, range [0.0 ~ 1.0]
            quality: JPEG type quality factor, take effect only if quant_tables is set to None
            quant_tables: set None to apply bulit-in tables, or [Y_TABLE, C_TABLE], [2, 8, 8]
            chroma_sub: chroma_sub_sampling method, avg pool by default
            range: image range, by default 255.

        Returns: y_coefficients, c_coefficients, quant_tables in [2, 8, 8]

        """
        b, c, h, w = x.shape
        x = x * range

        def rounding(x):
            if skip_table_rounding:
                return x
            else:
                return torch.round(x)

        if quant_tables is not None:
            y_t, c_t = rounding(quant_tables[0]), rounding(quant_tables[1])
        else:
            # Quality Factor control
            factor = quality_to_factor(quality)
            y_t, c_t = rounding(self.Y_TABLE * factor), rounding(self.C_TABLE * factor)

        # clamp the table
        y_t = torch.clamp(y_t, 1.0, np.iinfo(np.int16).max).to(x.device).float()
        c_t = torch.clamp(c_t, 1.0, np.iinfo(np.int16).max).to(x.device).float()

        # rgb 2 ycbcr
        ycbcr_x = self.color_cvt.rgb2ycbcr(x)
        y = ycbcr_x[:, 0, ...].unsqueeze(1)
        cbcr = ycbcr_x[:, 1:, ...]

        # chroma sub
        cbcr = chroma_sub(cbcr)
        # DCT
        y_coeffs = self.dct(y - 128.)  # B, 1, 8*8, H // 8, W // 8
        y_coeffs = y_coeffs.permute(0, 1, 3, 4, 2).reshape(-1, 8, 8)
        c_coeffs = self.dct(cbcr - 128.)
        c_coeffs = c_coeffs.permute(0, 1, 3, 4, 2).reshape(-1, 8, 8)

        # Quantization
        y_coeffs_quant = self.rounding(y_coeffs / y_t)
        c_coeffs_quant = self.rounding(c_coeffs / c_t)

        # Reshape
        y_coeffs_quant = y_coeffs_quant.reshape(b, 1, h // 8, w // 8, 8, 8) \
            .permute(0, 1, 2, 4, 3, 5).reshape(b, 1, h, w)

        c_coeffs_quant = c_coeffs_quant.reshape(b, 2, h // (8*scale_chroma_sub), w // (8*scale_chroma_sub), 8, 8) \
            .permute(0, 1, 2, 4, 3, 5).reshape(b, 2, h // scale_chroma_sub, w // scale_chroma_sub)

        return y_coeffs_quant, c_coeffs_quant, torch.stack([y_t, c_t], dim=0)

    def decompress(self, y_coeffs, c_coeffs, quant_tables=None,
                   chroma_up=default_chroma_up, scale_chroma_sub=2,
                   quality=50, range=255., **kwargs):
        """

        Args:
            y_coeffs: y channel quantized dct coefficients, [B, 1, H, W], range [0 ~ 255], uint8
            c_coeffs: cbcr channel quantized dct coefficients, [B, 2, H // 2, W // 2], range [0 ~ 255], uint8
            quant_tables: [Y_TABLE, C_TABLE], set None to apply bulit-in tables
            chroma_up: chroma_up_sampling method, repeat by default
            quality: JPEG type quality factor, take effect only if quant_tables is set to None
            range: image range, by default 255.

        Returns:

        """
        if quant_tables is not None:
            y_t, c_t = quant_tables[0], quant_tables[1]
        else:
            # Quality Factor control
            factor = quality_to_factor(quality)
            y_t, c_t = self.rounding(self.Y_TABLE * factor), self.rounding(self.C_TABLE * factor)

        if kwargs.get('clamp_table', True):
            y_t = torch.clamp(y_t, min=1.0, max=np.iinfo(np.uint16).max).float()
            c_t = torch.clamp(c_t, min=1.0, max=np.iinfo(np.uint16).max).float()
        else:
            y_t = y_t.float()
            c_t = c_t.float()

        b, _, h, w = y_coeffs.shape

        # Reshape
        y_coeffs = y_coeffs.reshape(b, 1, h // 8, 8, w // 8, 8) \
            .permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, 8)
        c_coeffs = c_coeffs.reshape(b, 2, h // (8*scale_chroma_sub), 8, w // (8*scale_chroma_sub), 8) \
            .permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, 8)

        # De-quantization
        y_coeffs_de_quant = y_coeffs * y_t  # B*C*nH*nW, 8, 8
        c_coeffs_de_quant = c_coeffs * c_t

        # iDCT
        y_coeffs_de_quant = y_coeffs_de_quant.reshape(b, 1, h // 8, w // 8, 64).permute(0, 1, 4, 2, 3)
        y_de_quant = self.idct(y_coeffs_de_quant) + 128

        c_coeffs_de_quant = c_coeffs_de_quant.reshape(b, 2, h // (8*scale_chroma_sub), w // (8*scale_chroma_sub), 64).permute(0, 1, 4, 2, 3)
        c_de_quant = self.idct(c_coeffs_de_quant) + 128.

        # chroma upsampling
        c_de_quant = chroma_up(c_de_quant)

        # ycbcr 2 rgb
        ycbcr = torch.cat([y_de_quant, c_de_quant], dim=1)
        rgb = self.color_cvt.ycbcr2rgb(ycbcr)

        # normalize
        rgb = rgb / range

        return rgb

    def get_TABLE(self, quality):
        factor = quality_to_factor(quality)
        y_t, c_t = self.rounding(self.Y_TABLE * factor), self.rounding(self.C_TABLE * factor)
        return torch.stack([y_t, c_t], dim=0)

    def save_jpeg(self, path, y_coeffs_quant, c_coeffs_quant, tables, scale_chroma_sub=2, coeffs_shape='bchw', validate=False):
        try:
            import torchjpeg.codec
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f'Failed to import torchjpeg library, skip saving {path}.')
            return 0
            # raise e

        if coeffs_shape == 'bchw':
            b, _, h, w = y_coeffs_quant.shape
        elif coeffs_shape == 'bchw64':
            b, _, h_, w_, _ = y_coeffs_quant.shape
            h, w = h_ * 8, w_ * 8

        assert b == 1
        fake_img = torch.zeros([3, h, w])
        dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(fake_img, 100,
                                                                                                          scale_chroma_sub,
                                                                                                          scale_chroma_sub)

        quantization = quantization * 0 + torch.stack([tables[0,], tables[1,], tables[1,]], dim=0).type(torch.ShortTensor)

        if coeffs_shape == 'bchw':
            y_coeffs_quant = y_coeffs_quant.reshape(b, 1, h // 8, 8, w // 8, 8)\
                                            .permute(0, 1, 2, 4, 3, 5).squeeze(0).type(torch.ShortTensor)
            c_coeffs_quant = c_coeffs_quant.reshape(b, 2, h // (8 * scale_chroma_sub), 8, w // (8 * scale_chroma_sub), 8)\
                                            .permute(0, 1, 2, 4, 3, 5).squeeze(0).type(torch.ShortTensor)
        elif coeffs_shape == 'bchw64':
            y_coeffs_quant = zig_zag_unflatten(y_coeffs_quant.reshape(b, -1, 64), (8, 8)) # b chw 8, 8
            y_coeffs_quant = y_coeffs_quant.reshape(b, 1, h_, w_, 8, 8).squeeze(0).type(torch.ShortTensor)
            c_coeffs_quant = zig_zag_unflatten(c_coeffs_quant.reshape(b, -1, 64), (8, 8)) # b chw 8, 8
            c_coeffs_quant = c_coeffs_quant.reshape(b, 2, h_//scale_chroma_sub, w_//scale_chroma_sub, 8, 8)\
                                           .squeeze(0).type(torch.ShortTensor)

        y_coeffs_quant = Y_coefficients * 0 + y_coeffs_quant
        c_coeffs_quant = CbCr_coefficients * 0 + c_coeffs_quant
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchjpeg.codec.write_coefficients(path, dimensions, quantization, y_coeffs_quant, c_coeffs_quant)

        if validate:
            dimensions_, quantization_, Y_coefficients_, CbCr_coefficients_ = torchjpeg.codec.read_coefficients(path)
            assert dimensions_.equal(dimensions) and quantization_.equal(quantization) and \
                    Y_coefficients_.equal(y_coeffs_quant) and CbCr_coefficients_.equal(c_coeffs_quant)
            print(path, 'pass validation')
        
        return 1
    
    def load_jpeg(self, path):
        try:
            import torchjpeg.codec
        except Exception as e:
            raise e
        
        dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
        
        return Y_coefficients, CbCr_coefficients, quantization[:2, ...].unsqueeze(0), (1, dimensions[0, 0].item(), dimensions[0, 1].item())