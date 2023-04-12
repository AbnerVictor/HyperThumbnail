import logging
import torch
from os import path as osp
from tqdm import tqdm

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path, cmd=None):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False, cmd=cmd)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in (opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    average_over_folders_dict = {}
    for test_loader in tqdm(test_loaders):
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        average_over_folders = model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        average_over_folders_dict[test_loader.dataset.opt['name']]=average_over_folders
    # total_avg_results is a dict: {
    #    'metric1': float,
    #    'metric2': float
    # }
    total_avg_results = {metric: 0 for metric in opt['val']['metrics'].keys()}
    for dataset_name, metrics in average_over_folders_dict.items():
        log_str = f'Validation {dataset_name}\n'
        for metric, value in metrics.items():
            if metric in total_avg_results.keys():
                log_str += f'\t # {metric}: {value:.4f}'
                total_avg_results[metric] += value
        logger = get_root_logger()
        logger.info(log_str)
    for metric in total_avg_results.keys():
        total_avg_results[metric] /= len(average_over_folders_dict)
    log_str = f'Average over all validation dataset\n'
    # average among folders
    for metric, value in total_avg_results.items():
        log_str += f'\t # {metric}: {value:.4f}'

    logger = get_root_logger()
    logger.info(log_str)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
