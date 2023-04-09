import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
except:
    pass

class EntropyModel(nn.Module):
    def __init__(self, K=1, N=64, hyperprior=False):
        super(EntropyModel, self).__init__()
        self.K = K
        self.N = N
        self.bottlenecks = nn.ModuleList([EntropyBottleneck(N) for i in range(K)])
        
    def compress(self, streams):
        assert len(streams) == self.K
        byte_arrays = []
        for i in range(self.K):
            self.bottlenecks[i].update()
            byte_array = self.bottlenecks[i].compress(streams[i])
            byte_arrays.append(byte_array)
        return byte_arrays

    def decompress(self, byte_arrays, shape):
        assert len(byte_arrays) == self.K
        streams = []
        for i in range(self.K):
            self.bottlenecks[i].update()
            byte_array = byte_arrays[i]
            stream = self.bottlenecks[i].decompress(byte_array, shape[i])
            streams.append(stream)
        return streams

    def forward(self, streams):
        assert len(streams) == self.K
        likelihoods = []
        streams_hat = []
        for i in range(self.K):
            stream_hat, likelihood = self.bottlenecks[i](streams[i])
            streams_hat.append(stream_hat)
            likelihoods.append(likelihood)
        return streams_hat, likelihoods

    def loss(self):
        aux_loss = 0
        for btn in self.bottlenecks:
            aux_loss += btn.loss()
        return aux_loss


class MLP(nn.Module):
    def __init__(self, in_chn, hidden_chn, out_chn, n_mlp=1, lr_mul=1, act=nn.Identity):
        super(MLP, self).__init__()
        self.first_linear = nn.Sequential(nn.Linear(in_chn, hidden_chn, bias=True),
                                          nn.LeakyReLU(inplace=False))

        layers = []

        for i in range(n_mlp):
            layers.append(nn.Sequential(nn.Linear(hidden_chn, hidden_chn, bias=True),
                                        nn.LeakyReLU(inplace=False)))

        self.layers = nn.Sequential(*layers)

        self.final_linear = nn.Linear(hidden_chn, out_chn, bias=True)

        self.act = act()

    def forward(self, x):
        x = self.first_linear(x)
        x = self.layers(x)
        x = self.final_linear(x)
        return self.act(x)


def noise_round(inputs):
    '''
        derivable rounding approximation from Scale Hyperprior
    '''
    half = float(0.5)
    noise = torch.empty_like(inputs).uniform_(-half, half)
    inputs = inputs + noise
    return inputs


class AvgLayer(nn.Module):
    def init(self):
        super(AvgLayer, self).__init__()

    def forward(self, x):
        # X: [B, C, N]
        return torch.mean(x, dim=-1, keepdim=True)


class Quant_Table_Predictor(nn.Module):
    def __init__(self, K=1, N=64, C=None, pool_type='mlp', act=nn.ReLU,
                 is_train=True, mean=None, var=None, inference_scalar = 1):
        super(Quant_Table_Predictor, self).__init__()
        self.K = K
        self.N = N
        self.MLPs = nn.ModuleList()
        for i in range(K):
            mlp = MLP(in_chn=N, hidden_chn=N * 3, out_chn=N, n_mlp=2, act=act)
            self.MLPs.append(mlp)

        self.POOLs = nn.ModuleList()
        for i in range(K):
            if pool_type == 'mlp':
                assert C is not None and len(C) == K
                mlp = MLP(in_chn=C[i], hidden_chn=C[i], out_chn=1, n_mlp=1, act=act)
                self.POOLs.append(mlp)
            elif pool_type == 'none':
                self.POOLs.append(nn.Identity())
            elif pool_type == 'avg':
                self.POOLs.append(AvgLayer())
            else:
                raise NotImplementedError(f'unsupported pool_type {pool_type}')

        if mean is None:
            mean = [torch.zeros(N) for i in range(K)]
        if var is None:
            var = [torch.ones(N) for i in range(K)]

        self.mean = mean
        self.var = var
        self.inference_scalar = inference_scalar
        if is_train:
            self.round = noise_round
        else:
            self.round = torch.round

    def forward(self, streams):
        '''

        Args:
            streams: K * [B, C, N]

        Returns: K * [B, N] in uint16

        '''
        assert len(streams) == self.K
        tables = []
        for i in range(self.K):
            stream = streams[i]
            mlp = self.MLPs[i]
            b, c, n = stream.shape

            # normalization
            mean = self.mean[i].reshape(1, -1, n).to(stream.device)
            var = self.var[i].reshape(1, -1, n).to(stream.device)
            c_ = var.shape[1]
            if c_ != 1:
                stream = ((stream.reshape(b, c_, -1, n) - mean.unsqueeze(2)) / var.unsqueeze(2)).reshape(b, c, n)
            else:
                stream = (stream - mean) / var

            table = mlp(stream)

            bias = 1.49 # to prevent a < 1 table value

            # pooling
            pool = self.POOLs[i]
            table = table.permute(0, 2, 1)
            table_pooled = pool(table).reshape(b, n) + bias

            table_pooled = table_pooled * self.inference_scalar
            table_pooled = self.round(table_pooled)

            # clampling table pooled
            tables.append(table_pooled)
        return tables


if __name__ == '__main__':
    fake_ten = torch.randn((2, 128, 64))
    fake_scaler = [torch.randn(1, 64), torch.randn(1, 64)]
    model = Quant_Table_Predictor(K=1, N=64, C=[128], mean=fake_scaler, var=fake_scaler, pool_type='mlp')

    tables = model([fake_ten])
    print(tables[0])
    print(tables[0].shape)
