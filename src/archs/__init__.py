import importlib
from os import path as osp
from basicsr.utils import scandir

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]

# import all the arch modules
_arch_modules = [importlib.import_module(f'src.archs.{file_name}') for file_name in arch_filenames]
