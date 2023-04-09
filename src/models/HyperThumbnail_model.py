import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
from collections import OrderedDict
import os
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.models import lr_scheduler as lr_scheduler


@MODEL_REGISTRY.register()
class HyperThumbnail_Model(SRModel):
    def __init__(self, opt):
        super(HyperThumbnail_Model, self).__init__(opt)
        self.scaled_degrade_lr = None
        self.scaled_lr = None
        self.bpp = -1
        self.compressed_lr = None
        self.is_dist = True if self.opt['num_gpu'] > 1 else False

    def init_training_settings(self):
        self.is_dist = True if self.opt['num_gpu'] > 1 else False

        self.net_g.train()
        train_opt = self.opt['train']
        # self.train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # set if fix decoder
        self.train_decoder = train_opt.get('train_decoder', True)
        if not self.train_decoder:
            if self.is_dist:
                self.net_g.module.fix_decoder()
            else:
                self.net_g.fix_decoder()

        table_ablation_opt = train_opt.get('table_ablation', None)
        if table_ablation_opt is not None:
            init_table_type = table_ablation_opt.get('type', None)
            quality = table_ablation_opt.get('quality', None)
            learnable = table_ablation_opt.get('learnable', True)
            init_table = self.net_g.resetJPEG(init_table_type, quality, learnable)
            logger = get_root_logger()
            logger.info(init_table)

        # define losses
        self.cri_pix = None
        self.cri_pix_lr = None
        self.cri_pix_degrade_lr = None
        self.cri_perceptual = None
        self.cri_perceptual_lr = None
        self.cri_pix_ccm_lr = None
        self.cri_freq = None
        self.cri_freq_lr = None
        self.cri_gan = None
        self.cri_pix_ccm_lr = None

        # IRN
        self.cri_ce = None

        # Rate_Loss
        self.cri_entropy = None

        self.gradient_clipping = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)

        if train_opt.get('pixel_opt_lr'):
            self.cri_pix_lr = build_loss(train_opt['pixel_opt_lr']).to(self.device)

        if train_opt.get('pixel_opt_degrade_lr'):
            self.cri_pix_degrade_lr = build_loss(train_opt['pixel_opt_degrade_lr']).to(self.device)

        if train_opt.get('pixel_opt_ccm_lr'):
            self.cri_pix_ccm_lr = build_loss(train_opt['pixel_opt_ccm_lr']).to(self.device)
            
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)

        if train_opt.get('perceptual_opt_lr'):
            self.cri_perceptual_lr = build_loss(train_opt['perceptual_opt_lr']).to(self.device)

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ce_opt'):
            self.cri_ce = build_loss(train_opt['ce_opt']).to(self.device)

        if train_opt.get('entropy_opt'):
            self.cri_entropy = True
            self.entropy_loss_weight = train_opt['entropy_opt'].get('loss_weight', 1.0)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # gradient clipping
        if train_opt.get('gradient_clipping'):
            self.gradient_clipping = train_opt['gradient_clipping']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        # self.scaler=None
        if train_opt.get('fp16', False):
            self.scaler = torch.cuda.amp.GradScaler()
        self.train_opt = train_opt
    
    def infer_save_jpeg(self, pad_gt, out_jpeg_path=None):
        with torch.no_grad():
            byte_arrays, side_infos = self.net_g.compress(pad_gt,
                                        round_streams_before_compression=True)
        self.scaled_lr = side_infos['scaled_lr']
        
        ### save jpeg
        stream_y, stream_c = side_infos['streams']
        tables = side_infos['tables'].squeeze(0)
        b, h, w = side_infos['shape']
        scale_chroma_sub = 2 if self.net_g.chroma_sub else 1
        stream_y = stream_y.reshape(b, 1, 64, h // 8, w // 8).permute(0, 1, 3, 4, 2)
        stream_c = stream_c.reshape(b, 2, 64, h // (8 * scale_chroma_sub), w // (8 * scale_chroma_sub)).permute(0, 1, 3, 4, 2)
        
        if self.net_g.losslessJPEG.save_jpeg(out_jpeg_path, stream_y, stream_c,\
                                    tables, scale_chroma_sub=scale_chroma_sub, coeffs_shape='bchw64', validate=False):
            bpp_jpeg = os.path.getsize(out_jpeg_path) * 8 / (b * h * w * self.opt['scale'] * self.opt['scale'])
            self.bpp = bpp_jpeg
            
            # load info from jpeg
            stream_y_hat, stream_c_hat, side_infos = self.net_g.load_jpeg(out_jpeg_path)
            
            with torch.no_grad():
                out, jpeg = self.net_g.decompress(None, side_infos, stream_y_hat=stream_y_hat, stream_c_hat=stream_c_hat,
                                                  trt=False, from_entropy_bottleneck_byte_arrays=False)
            
        else:
            bpp_jpeg = 0
            self.bpp = bpp_jpeg
        
            with torch.no_grad():
                out, jpeg = self.net_g.decompress(byte_arrays, side_infos, trt=False)
        
        self.scaled_degrade_lr = jpeg['rgb']
        self.compressed_lr = [jpeg['y_coeffs'], jpeg['c_coeffs'], jpeg['tables']]
        self.output = out
        
    def infer(self, input, **kwargs):
        self.output, self.scaled_lr, jpeg = self.net_g(input)
        self.scaled_degrade_lr = jpeg['rgb']
        self.ccm_lr = jpeg.get('idct_C_out', None)
        self.stream_likelihood = jpeg['stream_likelihood']
        self.compressed_lr = [jpeg['y_coeffs'], jpeg['c_coeffs'], jpeg['tables']]

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            if train_opt['scheduler'].get('milestones', None):
                if isinstance(train_opt['scheduler']['milestones'], list):
                    pass
                else:
                    step_size = train_opt['scheduler']['milestones']
                    milestones = [step_size * i for i in range(1, train_opt.get('total_iter') // step_size)]
                    train_opt['scheduler']['milestones'] = milestones
                    logger = get_root_logger()
                    logger.info(f'{scheduler_type} milestones: {milestones}')
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'ReduceLROnPlateau':
            for optimizer in self.optimizers:
                self.schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        fix_and_grad = train_opt.get('fix_and_grad', {'fix':[], 'grad':[]})

        for k, v in self.net_g.named_parameters():
            fix_flag = False
            for k_ in fix_and_grad['fix']:
                if k_ in k:
                    fix_flag = True
            if not fix_flag and not k.endswith(".quantiles"):
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_params = []
        for k, v in self.net_g.named_parameters():
            if k.endswith(".quantiles"):
                optim_params.append(v)
                logger = get_root_logger()
                logger.warning(f'Aux Params {k} will be optimized.')

        optim_type = train_opt['optim_aux'].pop('type')
        self.optimizer_aux = self.get_optimizer(optim_type, optim_params, **train_opt['optim_aux'])
        self.optimizers.append(self.optimizer_aux)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # with torch.cuda.amp.autocast(self.train_opt.get('fp16', False)):
        padded_gt, mod_pad_h, mod_pad_w = self.pad_input(self.gt)  # pad gt
        self.infer(padded_gt)
        self.crop_output(mod_pad_h, mod_pad_w)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        try:
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                # raise Exception('test loss')
                if torch.isnan(l_pix).sum() > 0:
                    raise Exception('cri_pix nan loss')
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            if self.cri_pix_lr:
                l_pix = self.cri_pix_lr(self.scaled_lr, self.lq)
                if torch.isnan(l_pix).sum() > 0:
                    raise Exception('cri_pix_lr nan loss')
                l_total += l_pix
                loss_dict['l_pix_lr'] = l_pix

            if self.cri_pix_degrade_lr:
                l_pix = self.cri_pix_degrade_lr(self.scaled_degrade_lr, self.lq)
                if torch.isnan(l_pix).sum() > 0:
                    raise Exception('cri_pix_degrade_lr nan loss')
                l_total += l_pix
                loss_dict['l_pix_degrade_lr'] = l_pix

            if self.cri_pix_ccm_lr:
                assert self.ccm_lr is not None
                l_pix = self.cri_pix_ccm_lr(self.scaled_lr - self.scaled_degrade_lr, self.ccm_lr)
                if torch.isnan(l_pix).sum() > 0:
                    raise Exception('cri_pix_degrade_lr nan loss')
                l_total += l_pix
                loss_dict['l_pix_ccm_lr'] = l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

            if self.cri_perceptual_lr:
                l_percep, l_style = self.cri_perceptual_lr(self.scaled_lr, self.lq)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep_lr'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style_lr'] = l_style

            # GAN loss
            if self.cri_gan:
                l_gan = self.cri_gan(self.output, self.gt)
                l_total += l_gan
                loss_dict['l_gan'] = l_gan

            if self.cri_ce:
                z = self.net_g.module.z if self.is_dist else self.net_g.z
                l_ce = self.cri_ce(z)
                l_total += l_ce
                loss_dict['l_ce'] = l_ce

            # Entropy loss
            if self.cri_entropy:
                b, _, h, w =  self.output.shape
                bpp_loss = torch.log(self.stream_likelihood).sum() / (-math.log(2) * (b*h*w))
                self.bpp = bpp_loss

                if self.entropy_loss_weight > 0.0:
                    l_total += self.entropy_loss_weight * bpp_loss
                    loss_dict['l_bpp'] = self.entropy_loss_weight * bpp_loss

                # aux
                self.optimizer_aux.zero_grad()
                aux_loss = self.get_bare_model(self.net_g).entropy_bottleneck.loss()
                loss_dict['l_aux'] = aux_loss
                aux_loss.backward()
                self.optimizer_aux.step()
            
        except Exception as e:
            logger = get_root_logger()
            logger.info(e)
            self.save(epoch=-1, current_iter=-1)
            np.save(osp.join(self.opt['path']['models'], f'lastinput.npy'), padded_gt.cpu().detach().numpy())
            quit()
            
        try:
            assert torch.isnan(l_total).sum() == 0
            if self.train_opt.get('fp16', False):
                self.scaler.scale(l_total).backward()
                self.scaler.unscale_(self.optimizer_g)
            else:
                l_total.backward()

            # gradient clipping
            if self.gradient_clipping:
                nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.gradient_clipping, norm_type=2) # TODO: check clip at ddp
            if self.train_opt.get('fp16', False):
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                self.optimizer_g.step()
            # TODO check reduce loss in ddp
            self.log_dict = self.reduce_loss_dict(loss_dict)
        except Exception as e:
            logger = get_root_logger()
            logger.info(e)
            raise e
            
        if self.bpp:
            self.log_dict['bpp'] = self.bpp

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_input(self, input):
        window_size = self.opt['network_g'].get('mod_pad', 64)  # 64/4 = 16  DCT table in Y Cb, Cr is 16X16
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = input.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        padded_gt = F.pad(input, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return padded_gt, mod_pad_h, mod_pad_w

    def crop_output(self, mod_pad_h, mod_pad_w):
        scale = self.opt.get('scale', 1)
        _, _, h, w = self.scaled_lr.shape
        self.scaled_lr = self.scaled_lr[:, :, :h - int(mod_pad_h / scale), :w - int(mod_pad_w / scale)]
        self.scaled_degrade_lr = self.scaled_degrade_lr[:, :, :h - int(mod_pad_h / scale), :w - int(mod_pad_w / scale)]

        _, _, h, w = self.output.shape
        self.output = self.output[:, :, :h - mod_pad_h, :w - mod_pad_w]

    def test(self, out_jpeg_path=None):
        self.net_g.eval()
        if self.opt['datasets'].get('val'):
            if self.opt['datasets']['val'].get('crop_val'):
                # pad the input according to window size
                if self.opt['datasets']['val'].get('gt_size'):
                    window_size = self.opt['datasets']['val'].get('gt_size')
                elif self.opt['datasets'].get('train'):
                    window_size = self.opt['datasets']['train'].get('gt_size')
                else:
                    window_size = 256
                scale = self.opt.get('scale', 1)
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = self.gt.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                gt = F.pad(self.gt, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                
                # patch to batch
                b, c, h, w = gt.shape
                h_patch_num = h // window_size
                w_patch_num = w // window_size

                gt = gt.view(b, c, h_patch_num, window_size, w_patch_num, window_size).permute(0, 2, 4, 1, 3,
                                                                                               5).reshape(-1, c,
                                                                                                          window_size,
                                                                                                          window_size)

                if self.opt['datasets']['val'].get('batch_size_per_gpu'):
                    batch_size_per_gpu = self.opt['datasets']['val'].get('batch_size_per_gpu')
                else:
                    batch_size_per_gpu = self.opt['datasets']['train'].get('batch_size_per_gpu')
        
                # if self.opt['datasets'].get('val'):
                #     if self.opt['datasets']['val'].get('crop_val'):
                scaled_degrade_lr = None
                scaled_lr = None
                output = None
                for i in range(int(np.ceil(gt.shape[0] / batch_size_per_gpu))):
                    start = i * batch_size_per_gpu
                    end = start + batch_size_per_gpu
                    end = end if end <= gt.shape[0] else gt.shape[0]                    
                    gt_ = gt[start:end, ...]
                    with torch.no_grad():
                        gt_, pad_h, pad_w = self.pad_input(gt_)
                        self.infer(gt_, out_jpeg_path=out_jpeg_path)
                        self.crop_output(pad_h, pad_w)

                    if scaled_lr is None:
                        scaled_degrade_lr = self.scaled_degrade_lr
                        scaled_lr = self.scaled_lr
                        output = self.output
                    else:
                        scaled_degrade_lr = torch.cat([scaled_degrade_lr, self.scaled_degrade_lr], dim=0)
                        scaled_lr = torch.cat([scaled_lr, self.scaled_lr], dim=0)
                        output = torch.cat([output, self.output], dim=0)
                    torch.cuda.empty_cache()

                self.scaled_degrade_lr = scaled_degrade_lr
                self.scaled_lr = scaled_lr
                self.output = output
                _, _, h_lr, w_lr = self.scaled_lr.shape
                self.scaled_lr = self.scaled_lr.reshape(b, h_patch_num, w_patch_num, c, h_lr, w_lr)\
                    .permute(0, 3, 1, 4, 2, 5).reshape(b, c, h_lr * h_patch_num, w_lr * w_patch_num)
                self.scaled_degrade_lr = self.scaled_degrade_lr.reshape(b, h_patch_num, w_patch_num, c, h_lr, w_lr)\
                    .permute(0, 3, 1, 4, 2, 5).reshape(b, c, h_lr * h_patch_num, w_lr * w_patch_num)
                self.output = self.output.reshape(b, h_patch_num, w_patch_num, c, window_size, window_size)\
                    .permute(0, 3, 1, 4, 2, 5).reshape(b, c, window_size * h_patch_num, window_size * w_patch_num)

                _, _, h, w = self.scaled_lr.shape
                self.scaled_lr = self.scaled_lr[:, :, 0:h - int(mod_pad_h / scale), 0:w - int(mod_pad_w / scale)]
                self.scaled_degrade_lr = self.scaled_degrade_lr[:, :, 0:h - int(mod_pad_h / scale), 0:w - int(mod_pad_w / scale)]

                _, _, h, w = self.output.shape
                self.output = self.output[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]                    
           
            else:
                with torch.no_grad():
                    gt, pad_h, pad_w = self.pad_input(self.gt)
                    if self.opt['val'].get('save_jpeg', False):
                        self.infer_save_jpeg(gt, out_jpeg_path=out_jpeg_path)
                    else:
                        self.infer(gt)
                    self.crop_output(pad_h, pad_w)

        if not self.opt['train'].get('fix_bn', False):
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'lq'):
            out_dict['lq'] = self.lq.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'scaled_lr'):
            out_dict['scaled_lr'] = self.scaled_lr.detach().cpu()
        if hasattr(self, 'scaled_degrade_lr'):
            out_dict['scaled_degrade_lr'] = self.scaled_degrade_lr.detach().cpu()
        if hasattr(self, 'compressed_lr'):
            out_dict['y_coeffs'] = self.compressed_lr[0]
            out_dict['c_coeffs'] = self.compressed_lr[1]
            out_dict['table'] = self.compressed_lr[2]
        if hasattr(self, 'bpp'):
            out_dict['bpp'] = self.bpp
        return out_dict

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        metric_data = None
        metric_data_lq = None
        metric_data_degrad_lq = None
        metric_data_bpp = None
        
        if with_metrics:
            self.metric_all_dict = {} # index with metric_all_dict['image']['metric']
            self.metric_all_dict['image_name'] = []
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
            self.metric_all_dict['hr'] = {metric: [] for metric in self.opt['val']['metrics'].keys()}
            
            # bpp
            self.metric_data_bpp = {'bpp': -1, 'cnt': 0}
            metric_data_bpp = -1

            if self.opt['val'].get('val_scaled_lq', False):
                self.metric_results_lq = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                metric_data_lq = dict()
                self.metric_all_dict['lq'] = {metric: [] for metric in self.opt['val']['metrics'].keys()}
                
            if self.opt['val'].get('val_scaled_degrad_lq', False):
                self.metric_results_degrad_lq = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                metric_data_degrad_lq = dict()
                self.metric_all_dict['degrad_lq'] = {metric: [] for metric in self.opt['val']['metrics'].keys()}

        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self.opt['is_train']:
                save_lr_pil_jpeg_path = osp.join(self.opt['path']['visualization'], img_name, 'torchjpeg_jpg',
                                    f'{img_name}_{current_iter}.jpg')
            else:
                save_lr_pil_jpeg_path = osp.join(self.opt['path']['visualization'], dataset_name, f'torchjpeg_jpg',
                            f'{img_name}_{self.opt["name"]}.jpg')
            self.test(save_lr_pil_jpeg_path)

            visuals = self.get_current_visuals()

            output_img = tensor2img([visuals['result']])
            if metric_data:
                metric_data['img'] = output_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            if 'lq' in visuals:
                lq_img = tensor2img([visuals['lq']])
                metric_data_lq['img2'] = lq_img
                if self.metric_results_degrad_lq:
                    metric_data_degrad_lq['img2'] = lq_img

            if 'scaled_lr' in visuals:
                scaled_lr_img = tensor2img([visuals['scaled_lr']])
            else:
                scaled_lr_img = None

            if 'scaled_degrade_lr' in visuals:
                scaled_degrade_lr_img = tensor2img([visuals['scaled_degrade_lr']])
            else:
                scaled_degrade_lr_img = None

            if 'bpp' in visuals:
                metric_data_bpp = visuals['bpp']
            else: 
                metric_data_bpp = -1

            # free memory
            if hasattr(self, 'lq'):
                del self.lq
            del self.gt
            del self.scaled_lr
            del self.output
            del self.compressed_lr
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, 'reconstruct_hr',
                                             f'{img_name}_{current_iter}.png')
                    save_scaled_lr_img_path = osp.join(self.opt['path']['visualization'], img_name, f'scaled_lr',
                                                       f'{img_name}_{current_iter}.png')
                    save_scaled_degrade_lr_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                               f'scaled_degrade_lr',
                                                               f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val'].get('suffix', None):
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'reconstruct_hr',
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_scaled_lr_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'scaled_lr',
                                                           f'{img_name}_{self.opt["name"]}.png')
                        save_scaled_degrade_lr_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'scaled_degrade_lr',
                                                           f'{img_name}_{self.opt["name"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'reconstruct_hr',
                                                 f'{img_name}_{self.opt["name"]}.png')
                        save_scaled_lr_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'scaled_lr',
                                                           f'{img_name}_{self.opt["name"]}.png')
                        save_scaled_degrade_lr_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'scaled_degrade_lr',
                                                           f'{img_name}_{self.opt["name"]}.png')

                imwrite(output_img, save_img_path)
                if scaled_lr_img is not None and self.metric_results_lq:
                    imwrite(scaled_lr_img, save_scaled_lr_img_path)
                if scaled_degrade_lr_img is not None and self.metric_results_degrad_lq:
                    imwrite(scaled_degrade_lr_img, save_scaled_degrade_lr_img_path)

            if with_metrics:
                self.metric_all_dict['image_name'].append(img_name)
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    opt__ = copy.deepcopy(opt_)
                    if opt_['type'] == 'calculate_bpp':
                        metric_data = {}
                        metric_data['img'] = save_lr_pil_jpeg_path
                        self.metric_results_degrad_lq[name] += calculate_metric(metric_data, opt__)
                    else:
                        metric_data['img'] = output_img
                        value = calculate_metric(metric_data, opt__)
                        self.metric_results[name] += value
                        self.metric_all_dict['hr'][name].append(value)
                        
                        logger = get_root_logger()
                        logger.info(f'{img_name} gt {name}: {value:.4f}')
                        if metric_data_lq is not None:
                            metric_data_lq['img'] = scaled_lr_img
                            value = calculate_metric(metric_data_lq, opt__)
                            self.metric_results_lq[name] += value
                            self.metric_all_dict['lq'][name].append(value)
                            
                            logger.info(f'{img_name} lq {name}: {value:.4f}')
                        if metric_data_degrad_lq is not None:
                            metric_data_degrad_lq['img'] = scaled_degrade_lr_img
                            value = calculate_metric(metric_data_degrad_lq, opt__)
                            self.metric_results_degrad_lq[name] += value
                            self.metric_all_dict['degrad_lq'][name].append(value)
                            
                            logger.info(f'{img_name} degrad {name}: {value:.4f}')
                # update bpp
                if metric_data_bpp != -1:
                    if self.metric_data_bpp['cnt'] == 0:
                        self.metric_data_bpp['bpp'] = metric_data_bpp
                    else:
                        self.metric_data_bpp['bpp'] += metric_data_bpp
                    self.metric_data_bpp['cnt'] += 1
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        # def print_stat(a): return f"shape={a.shape}, min={a.min():.2f}, median={a.median():.2f}, max={a.max():.2f}, var={a.var():.2f}, {a.flatten()[0]}"
        # for module_name in ["encoder", "decoder", "table_predictor", "entropy_bottleneck"]:
        #     module = getattr(self.net_g, module_name)
        #     p = list(module.named_parameters())[0][1]
        #     logger.info(f"parameters of {module_name}"+print_stat(p))

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                if self.metric_results_lq is not None:
                    self.metric_results_lq[metric] /= (idx + 1)
                if self.metric_results_degrad_lq is not None:
                    self.metric_results_degrad_lq[metric] /= (idx + 1)
            if self.metric_data_bpp['cnt'] != 0:
                self.metric_data_bpp['bpp'] /= self.metric_data_bpp['cnt']
            try:
                self.save_best_finetune_model(current_iter)
            except:
                pass
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            self.metric_results.update(self.metric_data_bpp)
            
            return self.metric_results
        
    def save_best_finetune_model(self, current_iter):
        if current_iter == 0:
            self.metric_results_init = self.metric_results
            self.metric_results_best = {'psnr_rgb':0.0}
        else:
            if self.metric_results['psnr_rgb'] > self.metric_results_best['psnr_rgb']:
                self.metric_results_best = self.metric_results
                self.save(-1, -1)
                logger = get_root_logger()            
                logger.info(f'Find better model {str(self.metric_results)}')
                logger.info('Save better model.')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        if self.metric_results_lq is not None:
            for metric, value in self.metric_results_lq.items():
                log_str += f'\t # {metric}_lq: {value:.4f}\n'
        if self.metric_results_degrad_lq is not None:
            for metric, value in self.metric_results_degrad_lq.items():
                log_str += f'\t # {metric}_degrad_lq: {value:.4f}\n'
        if self.metric_data_bpp is not None:
            bpp = self.metric_data_bpp['bpp']
            log_str += f'\t # bpp_degrad_lq: {bpp:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        import pandas as pd
        
        df_dict = {'name': self.metric_all_dict['image_name']}
        for sub_img in self.metric_all_dict:
            if isinstance(self.metric_all_dict[sub_img], dict):
                for metric in self.metric_all_dict[sub_img]:
                    df_dict[f'{sub_img}_{metric}'] = self.metric_all_dict[sub_img][metric]
        df = pd.DataFrame.from_dict(df_dict)
        csv_path = logger.handlers[1].baseFilename.replace(".log", ".csv")
        df.to_csv(csv_path)
        
        if tb_logger:
            tb_logger.add_scalar('lr', self.optimizer_g.param_groups[0]['lr'], current_iter)
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
            if self.metric_results_lq is not None:
                for metric, value in self.metric_results_lq.items():
                    tb_logger.add_scalar(f'metrics_lq/{metric}', value, current_iter)
            if self.metric_results_degrad_lq is not None:
                for metric, value in self.metric_results_degrad_lq.items():
                    tb_logger.add_scalar(f'metrics_degrad_lq/{metric}', value, current_iter)
            if self.metric_data_bpp is not None:
                bpp = self.metric_data_bpp['bpp']
                tb_logger.add_scalar(f'metrics_degrad_lq/bpp', bpp, current_iter)

