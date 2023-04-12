#%%
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
#%%
# modcrop
def resize_save_img_path(read_path, mod_save_path=None, lr_save_path=None):
    img = cv2.imread(read_path, cv2.IMREAD_UNCHANGED)
    
    mod = 24
    scale = 8
    h,w,c = img.shape
    h_mod = h//mod*mod
    w_mod = w//mod*mod
    img_mod = img[0:h_mod,0:w_mod,:]
    # img_resized = cv2.resize(img_mod, dsize=[h_mod//scale, w_mod//scale], interpolation=cv2.INTER_CUBIC)
    img_resized = cv2.resize(img_mod, dsize=[w_mod//scale, h_mod//scale], interpolation=cv2.INTER_CUBIC)
    if mod_save_path is not None:
        cv2.imwrite(mod_save_path+'/'+os.path.basename(read_path), img_mod)    
    # img_resized = cv2.resize(img_mod, dsize=[h_mod//scale, w_mod//scale])
    cv2.imwrite(lr_save_path+'/'+os.path.basename(read_path), img_resized)

def resize_save_img_folder(read_path_folder,mod_save_path, lr_save_path):
    # read_path_folder = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_sub'
    os.makedirs(mod_save_path, exist_ok=True)
    os.makedirs(lr_save_path, exist_ok=True)
    read_path_list = glob(read_path_folder+'/*.png')
    for read_path in tqdm(read_path_list):
        resize_save_img_path(read_path, mod_save_path, lr_save_path)  


#%%
if __name__ =="__main__":
    # read_path = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_sub/0001_s001.png'
    # save_path = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/LR/DIV2K_train_LR_bicubic/X8_sub'
    read_path_folder = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/Set14/GTmod12'
    mod_save_path = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/Set14/GTmod24'
    lr_save_path = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/Set14/LRbicx8'
    
    resize_save_img_folder(read_path_folder, mod_save_path, lr_save_path)
    # read_path_folder = '/home/cqiaa/fast_rescaling/Video-Enhancement-Playground/datasets/Image_Super_Resolution/Classic/DIV2K/HR/DIV2K_train_HR_sub'
    # read_path_list = glob(read_path_folder+'/*.png')
    # for read_path in tqdm(read_path_list):
    #     resize_save_img_path(read_path, save_path)








#%%



# def crop_dataset():

#     pass



# def crop_dataset_list(dataset_list):
#     for d in dataset_list:
#         crop_dataset(d)
#     pass


# def generate_bicubic_img():
#     # %# matlab code to genetate mod images, bicubic-downsampled images and
#     # %# bicubic_upsampled images

#     # %# set configurations
#     # # comment the unnecessary lines
#     input_folder = '../../datasets/Set5/original';
#     save_mod_folder = '../../datasets/Set5/GTmod12';
#     save_lr_folder = '../../datasets/Set5/LRbicx2';
#     # save_bic_folder = '';

#     mod_scale = 12;
#     up_scale = 2;

#     if exist('save_mod_folder', 'var')
#         if exist(save_mod_folder, 'dir')
#             disp(['It will cover ', save_mod_folder]);
#         else
#             mkdir(save_mod_folder);
#         end
#     end
#     if exist('save_lr_folder', 'var')
#         if exist(save_lr_folder, 'dir')
#             disp(['It will cover ', save_lr_folder]);
#         else
#             mkdir(save_lr_folder);
#         end
#     end
#     if exist('save_bic_folder', 'var')
#         if exist(save_bic_folder, 'dir')
#             disp(['It will cover ', save_bic_folder]);
#         else
#             mkdir(save_bic_folder);
#         end
#     end

#     idx = 0;
#     filepaths = dir(fullfile(input_folder,'*.*'));
#     for i = 1 : length(filepaths)
#         [paths, img_name, ext] = fileparts(filepaths(i).name);
#         if isempty(img_name)
#             disp('Ignore . folder.');
#         elseif strcmp(img_name, '.')
#             disp('Ignore .. folder.');
#         else
#             idx = idx + 1;
#             str_result = sprintf('%d\t%s.\n', idx, img_name);
#             fprintf(str_result);

#             # read image
#             img = imread(fullfile(input_folder, [img_name, ext]));
#             img = im2double(img);

#             # modcrop
#             img = modcrop(img, mod_scale);
#             if exist('save_mod_folder', 'var')
#                 imwrite(img, fullfile(save_mod_folder, [img_name, '.png']));
#             end

#             # LR
#             im_lr = imresize(img, 1/up_scale, 'bicubic');
#             if exist('save_lr_folder', 'var')
#                 imwrite(im_lr, fullfile(save_lr_folder, [img_name, '.png']));
#             end

#             # Bicubic
#             if exist('save_bic_folder', 'var')
#                 im_bicubic = imresize(im_lr, up_scale, 'bicubic');
#                 imwrite(im_bicubic, fullfile(save_bic_folder, [img_name, '.png']));
#             end
#         end
#     end
#     end

