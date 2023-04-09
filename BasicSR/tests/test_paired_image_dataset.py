import math
import os
import torchvision.utils

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler


def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'PairedImageDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_sub'
        opt['dataroot_lq'] = 'datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_train_LR_bicubic/X4_sub'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'meta_info_file':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'  # noqa:E501
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'  # noqa:E501
        opt['io_backend'] = dict(type='lmdb')

    opt['gt_size'] = 256
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 8
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 100
    opt['sorted'] = True
    os.makedirs('tmp', exist_ok=True)

    opt['world_size'] = 1
    opt['rank'] = 0
    dataset = build_dataset(opt)
    train_sampler = EnlargedSampler(dataset, opt['world_size'], opt['rank'], opt['dataset_enlarge_ratio'])
    data_loader = build_dataloader(dataset, opt, num_gpu=1, dist=opt['dist'], sampler=train_sampler, seed=10)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)
        torchvision.utils.save_image(lq, f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main()
