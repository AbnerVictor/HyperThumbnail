import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir


def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  IMWRITE_JPEG_QUALITY from 0 to 100.
            A higher value means a larger file size.
            Default: 90

        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR
            DIV2K_train_LR_bicubic/X2
            DIV2K_train_LR_bicubic/X3
            DIV2K_train_LR_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 96
    workspace_path = '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/'
    # HR images
    # /home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR
    # opt['input_folder'] = workspace_path+'datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR'
    # opt['save_folder'] = workspace_path+'datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_sub_JPEG'
    # opt['crop_size'] = 480
    # opt['step'] = 240
    # opt['thresh_size'] = 0
    # extract_subimages(opt)

    # LRx2 images
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    # opt['crop_size'] = 240
    # opt['step'] = 120
    # opt['thresh_size'] = 0
    # extract_subimages(opt)

    # LRx3 images
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    # opt['crop_size'] = 160
    # opt['step'] = 80
    # opt['thresh_size'] = 0
    # extract_subimages(opt)

    # LRx4 images
    # opt['input_folder'] = 'datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_train_LR_bicubic/X4'
    # opt['save_folder'] = 'datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_train_LR_bicubic/X4_sub_JPEG'
    # opt['crop_size'] = 120
    # opt['step'] = 60
    # opt['thresh_size'] = 0
    # datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_valid_LR_bicubic
    # opt['input_folder'] = workspace_path+'datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_valid_LR_bicubic/X4'
    # opt['save_folder']  = workspace_path+'datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_valid_LR_bicubic/X4_sub_JPEG'
    # /home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/Set14/LRbicx4
    # /home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/BSDS100/LR/bicubic_4x
    # datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_valid_LR_bicubic/X4
    
    input_folder_list = [
    # '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/kodak/LRbicx4',
    '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/Set14/LRbicx4',
    '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/Set5/LRbicx4',
    '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/BSDS100/LR/bicubic_4x',
    '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/urban100/LR/bicubic_4x',
    '/home/chenyangqi/disk1/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_valid_LR_bicubic/X4'
    ]
    for input_folder in input_folder_list:
        opt['input_folder'] = input_folder
        # opt['save_folder']  = workspace_path+'datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_valid_LR_bicubic/X4_JPEG'
        opt['save_folder']  = opt['input_folder']+'_JPEG'+str(opt['compression_level'])

        # input_path = '/home/chenyangqi/disk1/fast_rescaling/CAR/result_div2k'
        # opt['input_folder'] = input_path
        # Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_CAR
        # opt['save_folder']  = workspace_path+'datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_CAR'
        # opt['crop_size'] = None
        # opt['step'] = 100000
        # opt['thresh_size'] = -1
        extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists.')
        # sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    # crop_size = opt['crop_size']
    # step = opt['step']
    # thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    print(img_name, extension)
    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # h, w = img.shape[0:2]
    # if crop_size is None:
        # crop_size = h
    # h_space = np.arange(0, h - crop_size + 1, step)
    # if h - (h_space[-1] + crop_size) > thresh_size:
        # h_space = np.append(h_space, h - crop_size)
    # if crop_size is None:
    #     crop_size = w
    # w_space = np.arange(0, w - crop_size + 1, step)
    # if w - (w_space[-1] + crop_size) > thresh_size:
    #     w_space = np.append(w_space, w - crop_size)

    # index = 0
    # for x in h_space:
        # for y in w_space:
            # index += 1
    cropped_img = img
    cropped_img = np.ascontiguousarray(cropped_img)
    # print(osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'))
    cv2.imwrite(
        osp.join(opt['save_folder'], f'{img_name}.jpg'), cropped_img,
        [cv2.IMWRITE_JPEG_QUALITY, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
