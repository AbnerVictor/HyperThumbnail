#%%
import torch
import torch.nn as nn
import numpy as np
import time

from basicsr.archs.edsr_arch import EDSR
from basicsr.utils.registry import ARCH_REGISTRY
from src.ops.RDBUnet_0812 import RDBUnet
from src.ops.coeff_edsr import Coeff_EDSR
from src.ops.compression_modules import EntropyModel, Quant_Table_Predictor
from src.ops.JPEG_Utils.DiffJPEG import DiffJPEG, JPEG_Y_TABLE, JPEG_C_TABLE
from src.ops.JPEG_Utils.dot_utils import streamFlatten, zig_zag_flatten, zig_zag_unflatten
from src.ops.JPEG_Utils.utils import skip_chroma_sub, default_chroma_sub, default_chroma_up

LOSSLESS_Y = np.ones((8, 8))
LOSSLESS_C = np.ones((8, 8))

@ARCH_REGISTRY.register()
class HyperThumbnail(nn.Module):
    def __init__(self, scale=1, patch_size=256, table='ones', quality=100, enc_num_feat=64,
                 quant_type='bypass', optim_table=False, chroma_sub=False, stream_shape='bchw', down_layer=2, 
                 down_by_dense=False, slim_channel=False, predict_quant_table=False, is_train=True, 
                 decoder_opt=None, coe_decoder_opt=None, **kwargs):
        super(HyperThumbnail, self).__init__()
        self.scale = scale
        self.encoder = RDBUnet(scale=scale, input_chn=3, out_chn=3, num_feat=enc_num_feat, 
                               down_layer=down_layer, down_by_dense=down_by_dense, slim_channel=slim_channel)
        self.coefficient_decoder = Coeff_EDSR(**coe_decoder_opt)
        assert chroma_sub == False, "Current coefficient_decoder Implementation only support no chroma subsampling"
        self.decoder = EDSR(**decoder_opt)

        Y_tab = LOSSLESS_Y
        C_tab = LOSSLESS_C

        self.chroma_sub = chroma_sub
        self.quality = quality

        if table == 'JPEG':
            Y_tab = JPEG_Y_TABLE
            C_tab = JPEG_C_TABLE

        self.losslessJPEG = DiffJPEG(quant_type=quant_type, optimize_table=optim_table, init_table=(Y_tab, C_tab))

        # factorization
        self.embedding_length = (8 ** 2)
        self.entropy_bottleneck = EntropyModel(K=2, N=self.embedding_length)
        self.stream_shape = stream_shape

        #
        patch_size = patch_size // scale
        self.predict_quant_table = predict_quant_table
        chroma_patch = 8 if not self.chroma_sub else 16
        if predict_quant_table:
            self.table_predictor = Quant_Table_Predictor(K=2, N=self.embedding_length,
                                                         C=[((patch_size // 8) ** 2), ((patch_size // chroma_patch) ** 2) * 2],
                                                         pool_type=kwargs.get('pool_type', 'mlp'), is_train=is_train,
                                                         inference_scalar = kwargs.get('inference_scalar', 1.0))

    def compress(self, x, **kwargs):    
        if self.scale > 1:
            emb = self.encoder(x)
        else:
            emb = x
        scale_chroma_sub = 2 if self.chroma_sub else 1

        emb_ = torch.clamp(emb, 0.0, 1.0)
        y_coeffs, c_coeffs, dummy_tables_ = self.losslessJPEG.compress(emb_, quality=self.quality, skip_table_rounding=True,
                                                                       chroma_sub=default_chroma_sub if self.chroma_sub else skip_chroma_sub,
                                                                       scale_chroma_sub=scale_chroma_sub)

        # estimate rate
        b, _, h, w = y_coeffs.shape
        stream_y = streamFlatten(y_coeffs)
        stream_c = streamFlatten(c_coeffs)

        # estimate tables
        if self.predict_quant_table:
            tables_pred = self.table_predictor([stream_y, stream_c])
            table_y, table_c = tables_pred[0].unsqueeze(1), tables_pred[1].unsqueeze(1)  # b, 1, 64
            
            y_t = zig_zag_unflatten(table_y.detach().clone(), (8, 8))
            c_t = zig_zag_unflatten(table_c.detach().clone(), (8, 8))
            tables = torch.cat([y_t, c_t], dim=1)
            stream_y = stream_y / table_y
            stream_c = stream_c / table_c

        else:
            tables = dummy_tables_

        h_, w_ = h // 8, w // 8
        stream_y = stream_y.reshape(b, h_, w_, -1).permute(0, 3, 1, 2)
        stream_c = stream_c.reshape(b * 2, h_ // scale_chroma_sub, w_ // scale_chroma_sub, -1).permute(0, 3, 1, 2)

        if kwargs.get('round_streams_before_compression', False):
            stream_y = torch.round(stream_y)
            stream_c = torch.round(stream_c)

        byte_arrays = self.entropy_bottleneck.compress([stream_y, stream_c]) # # b, c, h, w ->

        byte_cnt = 0
        for byte_array in byte_arrays:
            for byte_array_ in byte_array:
                byte_cnt += len(byte_array_)

        bpp = byte_cnt * 8 / (x.shape[0] * x.shape[2] * x.shape[3])

        return byte_arrays, {'streams': [stream_y, stream_c], 'tables': tables, 'shape': (b, h, w), 'bpp': bpp, 'scaled_lr': emb_}

    def decompress(self, byte_arrays, side_info, **kwargs):
        b, h, w = side_info['shape']
        tables = side_info['tables']

        scale_chroma_sub = 2 if self.chroma_sub else 1
        h_, w_ = h // 8, w // 8

        if kwargs.get('from_entropy_bottleneck_byte_arrays', True):
            streams_hat = self.entropy_bottleneck.decompress(byte_arrays, shape=[(h_, w_), (h_ // scale_chroma_sub, w_ // scale_chroma_sub)])
            stream_y_hat = streams_hat[0].permute(0, 2, 3, 1).reshape(b, -1, self.embedding_length)
            stream_c_hat = streams_hat[1].permute(0, 2, 3, 1).reshape(b, -1, self.embedding_length)
        else:
            stream_y_hat = kwargs.get('stream_y_hat', None).permute(0, 1, 3, 2, 4).reshape(b, -1, h, w)
            stream_c_hat = kwargs.get('stream_c_hat', None).permute(0, 1, 3, 2, 4).reshape(b, -1, h, w)
            
            stream_y_hat = streamFlatten(stream_y_hat).reshape(b, -1, self.embedding_length)
            stream_c_hat = streamFlatten(stream_c_hat).reshape(b, -1, self.embedding_length)
            
        if self.predict_quant_table:
            y_t, c_t = tables[:, :1, ...], tables[:, 1:, ...]
            table_y, table_c = zig_zag_flatten(y_t), zig_zag_flatten(c_t)

            stream_y_hat = stream_y_hat * table_y
            stream_c_hat = stream_c_hat * table_c

        y_coeffs_hat = streamFlatten(stream_y_hat, inverse=True, size=(h, w))
        c_coeffs_hat = streamFlatten(stream_c_hat, inverse=True, size=(h // scale_chroma_sub, w // scale_chroma_sub))

        y_coeffs_patch2depth = stream_y_hat.reshape(b, h // 8,  w // 8, -1).permute(0, 3, 1, 2)
        c_coeffs_patch2depth = stream_c_hat.reshape(b, 2, h // (8*scale_chroma_sub), w // (8*scale_chroma_sub), -1).permute(0, 1, 4, 2, 3) \
            .reshape(b, -1, h // (8*scale_chroma_sub), w // (8*scale_chroma_sub))
        
        dummy_tables_ = torch.ones((2, 8, 8), device=y_coeffs_hat.device)
        decomp_emb = self.losslessJPEG.decompress(y_coeffs_hat, c_coeffs_hat, dummy_tables_,
                                                  chroma_up=default_chroma_up if self.chroma_sub else skip_chroma_sub,
                                                  scale_chroma_sub=scale_chroma_sub)
        if kwargs.get('upsample', True):
            if not kwargs.get('trt', False):
                if self.scale > 1:
                    ccm_out = self.coefficient_decoder(torch.cat([y_coeffs_patch2depth, c_coeffs_patch2depth], dim=1))
                    out = self.decoder(torch.cat([decomp_emb, ccm_out], dim=1))
                else:
                    out = decomp_emb
            else:
                try:
                    from torch2trt import torch2trt
                    print('decoder to trt')
                    dec_trt = torch2trt(self.decoder.cuda(), [torch.zeros_like(decomp_emb.cuda())], fp16=kwargs.get('trt_fp16', True))
                    print('conversion done')
                    out = dec_trt(decomp_emb.cuda()).cpu()
                except Exception as e:
                    raise(e)
        else:
            out = decomp_emb

        return out, {'y_coeffs': y_coeffs_hat, 'c_coeffs': c_coeffs_hat, 'tables': tables, 'rgb': decomp_emb}


    def load_jpeg(self, path, device='cuda'):
        stream_y, stream_c, tables, shape = self.losslessJPEG.load_jpeg(path)       
        return stream_y.to(device), stream_c.to(device), {'tables': tables.to(device).float(), 'shape': shape}

    def forward(self, x, **kwargs):
        if self.scale > 1:
            emb = self.encoder(x) # N=16, C=3, H=256, W=256 -> 16,3,64,64
        else:
            emb = x

        scale_chroma_sub = 2 if self.chroma_sub else 1

        emb_ = torch.clamp(emb, 0.0, 1.0)
        y_coeffs, c_coeffs, dummy_tables_ = self.losslessJPEG.compress(emb_, quality=self.quality, skip_table_rounding=True,
                                                                       chroma_sub=default_chroma_sub if self.chroma_sub else skip_chroma_sub,
                                                                       scale_chroma_sub=scale_chroma_sub)
        # estimate rate
        b, _, h, w = y_coeffs.shape
        stream_y = streamFlatten(y_coeffs) # HW to zigzap; 16, 1, 64, 64 -> 16, 64, 64; floating
        stream_c = streamFlatten(c_coeffs) # HW to zigzap; 16, 2, 64, 64 -> 16, 64, 64; floating

        # estimate tables
        if self.predict_quant_table:
            tables_pred = self.table_predictor([stream_y, stream_c])
            table_y, table_c = tables_pred[0].unsqueeze(1), tables_pred[1].unsqueeze(1)  # b, 1, 64
            y_t = zig_zag_unflatten(table_y.detach().clone(), (8, 8))
            c_t = zig_zag_unflatten(table_c.detach().clone(), (8, 8))
            tables = torch.cat([y_t, c_t], dim=1)

            stream_y = stream_y / table_y
            stream_c = stream_c / table_c
        else:
            tables = dummy_tables_

        if self.stream_shape == 'bc': # bhw, c
            stream_y = stream_y.reshape(-1, self.embedding_length)
            stream_c = stream_c.reshape(-1, self.embedding_length)
        elif self.stream_shape == 'bchw': # b, c, hw
            h_, w_ = h // 8, w // 8
            stream_y = stream_y.reshape(b, h_, w_, -1).permute(0, 3, 1, 2)
            stream_c = stream_c.reshape(b * 2, h_ // scale_chroma_sub, w_ // scale_chroma_sub, -1).permute(0, 3, 1, 2)

        streams_hat, likelihoods = self.entropy_bottleneck([stream_y, stream_c])
        if scale_chroma_sub != 1:
            # cat likelihoods
            stream_likelihood = torch.cat([torch.flatten(likelihoods[0].permute(1, 0, 2, 3), start_dim=1),
                                           torch.flatten(likelihoods[1].permute(1, 0, 2, 3), start_dim=1)], dim=1)
        else:
            stream_likelihood = torch.cat(likelihoods, dim=0) # [3072, 64] in [0, 0.025]

        if self.stream_shape == 'bc':
            stream_y_hat = streams_hat[0].reshape(b, -1, self.embedding_length) # [16, 64, 64]
            stream_c_hat = streams_hat[1].reshape(b, -1, self.embedding_length)
        elif self.stream_shape == 'bchw':
            stream_y_hat = streams_hat[0].permute(0, 2, 3, 1).reshape(b, -1, self.embedding_length)
            stream_c_hat = streams_hat[1].permute(0, 2, 3, 1).reshape(b, -1, self.embedding_length)

        if self.predict_quant_table:
            stream_y_hat = stream_y_hat * table_y
            stream_c_hat = stream_c_hat * table_c

        y_coeffs_hat = streamFlatten(stream_y_hat, inverse=True, size=(h, w))
        c_coeffs_hat = streamFlatten(stream_c_hat, inverse=True, size=(h // scale_chroma_sub, w // scale_chroma_sub))

        decomp_emb = self.losslessJPEG.decompress(y_coeffs_hat, c_coeffs_hat, dummy_tables_,
                                                  chroma_up=default_chroma_up if self.chroma_sub else skip_chroma_sub,
                                                  scale_chroma_sub=scale_chroma_sub)
        y_coeffs_patch2depth = stream_y_hat.reshape(b, h // 8,  w // 8, -1).permute(0, 3, 1, 2)
        c_coeffs_patch2depth = stream_c_hat.reshape(b, 2, h // (8*scale_chroma_sub), w // (8*scale_chroma_sub), -1).permute(0, 1, 4, 2, 3) \
            .reshape(b, -1, h // (8*scale_chroma_sub), w // (8*scale_chroma_sub))
        if self.scale > 1:
            ccm_out = self.coefficient_decoder(torch.cat([y_coeffs_patch2depth, c_coeffs_patch2depth], dim=1))
            out = self.decoder(torch.cat([decomp_emb, ccm_out], dim=1)) # [16, 3, 64, 64]) -> [16, 3, 256, 256]
        else:
            out = decomp_emb

        return out, emb, {'y_coeffs': y_coeffs_hat, 'c_coeffs': c_coeffs_hat, 'tables': tables, 'rgb': decomp_emb,
                          'stream_likelihood': stream_likelihood}

#%%
if __name__ == '__main__':
    import os

    network_g_dict = {
        "type": "HyperThumbnail",
        "scale": 4,
        "patch_size": 256,
        "enc_num_feat": 64,
        "norm": "none",
        "quant_type": "bypass",
        "predict_quant_table": True,
        "pool_type": "avg",
        "chroma_sub": False,
        "is_train": False,
        'scale_pred_table': 'ones',
        'inference_scalar': 1.0,

        "decoder_opt": {
            "num_in_ch": 27,
            "num_out_ch": 3,
            "num_feat": 24,
            "num_block": 16,
            "upscale": 4,
            "res_scale": 1,
            "img_range": 1.0,
            "rgb_mean": [0.4488, 0.4371, 0.4040],
            "input_norm": False
            },
        
        "coe_decoder_opt": {
            "num_in_ch": 192,
            "num_out_ch": 24,
            "num_feat": 24,
            "num_block": 16,
            "upscale": 8,
            "res_scale": 1,
            "out_range": 10.0,
            "mean": None
        }}
    model = HyperThumbnail(**network_g_dict)

    fake_input = torch.randn((1, 3, 256, 256)).float()
    out, emb, jpeg = model(fake_input)
    
    # print(jpeg['tables'])
    def print_stat(a): print(f"shape={a.shape}, min={a.min():.2f}, median={a.median():.2f}, max={a.max():.2f}, var={a.var():.2f}, {a.flatten()[0]}")
    print_stat(fake_input)
    print_stat(emb)

