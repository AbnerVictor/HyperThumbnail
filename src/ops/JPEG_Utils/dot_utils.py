import torch
import numpy as np


def dct_matrix(window_size):
    i = torch.arange(0, window_size, dtype=torch.float)
    j = i.clone()
    freq_func = lambda x: np.pi * (2 * x + 1) / (2 * window_size)

    ij = torch.stack(torch.meshgrid(i, j)).view(-1, window_size, window_size)
    ij = torch.stack((torch.flatten(ij[0, ...]), torch.flatten(ij[1, ...]))).permute(1, 0)

    coef_1 = torch.ones_like(ij[:, 0]) * np.sqrt(1 / window_size)
    coef_2 = torch.ones_like(ij[:, 0]) * np.sqrt(2 / window_size)
    coef = torch.where(ij[:, 0] == 0, coef_1, coef_2)

    dct_mat = coef * torch.cos(ij[:, 0] * freq_func(ij[:, 1]))
    return dct_mat.view(window_size, window_size)


def dct_weights(window_size):
    p = 1
    # a = 1 / (torch.norm(b, embed_dim=2) ** p).view(1, -1)
    # a[0, 0] = 1

    j = torch.arange(0, window_size, dtype=torch.float)
    uv = torch.stack(torch.meshgrid(j, j)).view(-1, window_size, window_size)
    uv = torch.stack((torch.flatten(uv[0, ...]), torch.flatten(uv[1, ...]))).transpose(-2, -1).unsqueeze(-1)

    coef_1 = torch.ones_like(uv) * np.sqrt(1 / window_size)
    coef_2 = torch.ones_like(uv) * np.sqrt(2 / window_size)
    coef = torch.where(uv == 0, coef_1, coef_2).squeeze(-1)
    uv = uv.squeeze(-1)

    i = torch.arange(0, window_size, dtype=torch.float)
    i = np.pi * (i + 0.5) / window_size
    ij = torch.stack(torch.meshgrid(i, i)).view(-1, window_size, window_size)
    ij = torch.stack((torch.flatten(ij[0, ...]), torch.flatten(ij[1, ...]))).permute(1, 0)

    w = torch.cos(ij[:, 0].unsqueeze(-1) @ uv[:, 0].unsqueeze(-1).transpose(-2, -1))
    w *= torch.cos(ij[:, 1].unsqueeze(-1) @ uv[:, 1].unsqueeze(-1).transpose(-2, -1))

    coef_u = torch.ones(window_size ** 2, 1) @ coef[:, 0].unsqueeze(-1).transpose(-2, -1)
    coef_v = torch.ones(window_size ** 2, 1) @ coef[:, 1].unsqueeze(-1).transpose(-2, -1)
    w = w * coef_u * coef_v
    return w


def dct(x, dct_mat):
    return dct_mat @ x @ dct_mat.transpose(-2, -1)


def inv_dct(spectrum, dct_mat):
    res = dct_mat.transpose(-2, -1) @ spectrum
    res = res @ dct_mat
    return res


def cas(theta):
    return torch.cos(theta) + torch.sin(theta)


def dht_mat_1D(size):
    i = torch.arange(size)
    ids = torch.stack(torch.meshgrid(i, i)).view(-1, size, size)
    H = cas(2 * np.pi * ids[0, ...] * ids[1, ...] / size)
    return H / np.sqrt(size)


def dht_matrix(window_size):
    h, w = window_size
    H_h = dht_mat_1D(h)
    H_w = dht_mat_1D(w)
    return torch.stack((H_h, H_w))


def dht(x, dht_mat):
    H_h, H_w = dht_mat[0, ...], dht_mat[1, ...]
    res = H_h @ x @ H_w
    return res


def zig_zag_flatten(x):
    b, c, h, w = x.shape
    index = torch.arange(h * w).reshape(h, w)
    index = torch.flip(index, dims=[1])
    offsets = np.arange(-(h - 1), w)[::-1]

    flat_index = []
    for i in range(len(offsets)):
        dia = torch.diagonal(index, offset=offsets[i], dim1=0, dim2=1)
        if i % 2 == 0:
            dia = torch.flip(dia, dims=[-1])
        flat_index.append(dia)
    flat_index = torch.cat(flat_index, dim=-1)

    return torch.flatten(x, start_dim=2)[:, :, flat_index]


def zig_zag_unflatten(x, size):
    h, w = size[0], size[1]
    b, c, n = x.shape

    assert n == h * w

    index = torch.arange(h * w).reshape(1, 1, h, w)
    flat_index = zig_zag_flatten(index)
    sort_index, sort_indices = torch.sort(flat_index, dim=-1)
    x = x[:, :, sort_indices].reshape(b, c, h, w)
    return x


# if __name__ == '__main__':
#     x = torch.rand((2, 2, 3, 4))
#     print(x)
#
#     flat = zig_zag_flatten(x)
#     print(flat)
#
#     x_ = zig_zag_unflatten(flat, (3, 4))
#     print(x_)

def patch_to_channel(stream, patch_size=8):
    """ bchw To bchw88

    Args:
        stream (bchw): frequency is embedded in spatial patch
        patch_size (int, optional): _description_. Defaults to 8.

    Returns:
        bchw88: 
    """
    b, c, h, w = stream.shape
    stream = stream \
        .reshape(b, -1, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 1, 2, 4, 3, 5)
    return stream
def streamFlatten(stream, inverse=False, size=(8, 8), patch_size=8, difference_DC=True, JPEG_like_stream=False):
    '''
    The H, W dimension of input variable Stream are embedded with 64 frequency as default order.
    
    First reshape it to b, c, h/8, w/8, 8, 8。
    
    Then  reshape it to b, c*h/8*w/8, 8, 8 to squeeze spatial dimension.
    
    Finally, turn the last 8,8 from 2d dimension to 1d zig zag order
    
    Args:
        stream: B, C, H, W
        inverse: False to flat, True to unflat

    Returns: B, C * (H * W // patch_size ** 2), patch_size ** 2

    '''

    if not inverse:
        b, c, h, w = stream.shape
        stream = stream \
            .reshape(b, -1, h // patch_size, patch_size, w // patch_size, patch_size) \
            .permute(0, 1, 2, 4, 3, 5)

        b, c, h, w, _, _ = stream.shape
        stream_flat = zig_zag_flatten(stream.reshape(b, -1, 8, 8))

        if JPEG_like_stream:
            dc = stream_flat[:, :, 0:1]
            ac = stream_flat[:, :, 1:]

            # difference of dc
            if difference_DC:
                dc = dc.reshape(b, c, -1)
                dc0 = dc[:, :, 0:1]  # b, c, 1
                dc1 = dc[:, :, 1:] - dc[:, :, :-1]
                dc = torch.cat([dc0, dc1], dim=-1).reshape(b, -1)
            return torch.cat([dc.reshape(b, -1), ac.reshape(b, -1)], dim=-1)

        return stream_flat
    else:
        h, w = size
        if JPEG_like_stream:
            b, n = stream.shape
            n = n // patch_size ** 2
            c = n // (h * w // patch_size**2)

            dc = stream[:, :n].reshape(b, n, 1)
            ac = stream[:, n:].reshape(b, n, -1)

            # difference of dc
            if difference_DC:
                dc = dc.reshape(b, c, -1)
                dc0 = dc[:, :, 0:1]  # b, c, 1
                dc1 = torch.stack([torch.sum(dc[:, :, :i], dim=-1) for i in range(2, dc.shape[-1] + 1)], dim=2)
                dc = torch.cat([dc0, dc1], dim=-1).reshape(b, -1, 1)

            stream = zig_zag_unflatten(torch.cat([dc, ac], dim=-1), size=(patch_size, patch_size))

        else:
            b, n, _ = stream.shape # [1, 384, 64]
            c = n // (h * w // patch_size**2)
            # coefficient_b64hw = stream.reshape(b, , patch_size**2)
            stream = zig_zag_unflatten(stream, size=(patch_size, patch_size))

        stream = stream.reshape(b, c, h // patch_size, w // patch_size, patch_size, patch_size)\
                        .permute(0, 1, 2, 4, 3, 5)\
                        .reshape(b, c, h, w)
        return stream