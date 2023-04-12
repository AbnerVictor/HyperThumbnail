import os.path as osp
from basicsr.train import train_pipeline

import src.ops
import src.archs
import src.models
import src.scripts
import src.metrics

def train(runtime_root=None, cmd=None):
    if runtime_root is None:
        runtime_root = osp.abspath(osp.join(__file__, osp.pardir, '..', '..'))
    print(f'current root path: {runtime_root}, with cmd {cmd}')
    return train_pipeline(runtime_root, cmd=cmd)
