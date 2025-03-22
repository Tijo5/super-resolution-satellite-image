import os
import torch
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import USMSharp, get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from ssr.losses import build_loss
from ssr.metrics import calculate_metric

@MODEL_REGISTRY.register()
class SSRESRGANModel(SRGANModel):
    def __init__(self, opt):
        super(SSRESRGANModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device) if train_opt.get('gan_opt') else None
        self.ssim_loss = build_loss(train_opt['ssim_opt']).to(self.device) if train_opt.get('ssim_opt') else None
        self.clip_sim = build_loss(train_opt['clip_opt']).to(self.device) if train_opt.get('clip_opt') else None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.setup_optimizers()
        self.setup_schedulers()

    @torch.no_grad()
    def feed_data(self, data):
        self.lr = data['lr'].to(self.device).float()/255
        self.lq = self.lr
        if 'hr' in data:
            self.gt = data['hr'].to(self.device).float()/255
            self.gt_usm = self.usm_sharpener(self.gt)

        self.old_hr = data.get('old_hr', None)
        if self.old_hr is not None:
            self.old_hr = self.old_hr.to(self.device).float()/255

        self.feed_disc_lr = self.opt.get('feed_disc_lr', False)

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # Upsample the low-res input images to that of the ground truth so they can be stacked.
        lr_resized = F.interpolate(self.lr, scale_factor=4)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lr)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # SSIM loss
            if self.ssim_loss:
                l_g_ssim = self.ssim_loss(self.output, percep_gt)
                l_g_total += l_g_ssim
                loss_dict['l_g_ssim'] = l_g_ssim

            # Discriminator input construction (clean & safe)
            disc_input_parts = [self.output]
            if self.feed_disc_lr:
                disc_input_parts.append(lr_resized)
            if self.old_hr is not None:
                disc_input_parts.append(self.old_hr)
            disc_input = torch.cat(disc_input_parts, dim=1)

            # gan loss
            fake_g_pred = self.net_d(disc_input)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # clip similarity loss if used
            if self.clip_sim:
                l_clip_sim = self.clip_sim(self.output, l1_gt)
                loss_dict['l_clip_sim'] = l_clip_sim
                l_g_total += l_clip_sim

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        # Real discriminator input
        real_disc_input_parts = [gan_gt]
        if self.feed_disc_lr:
            real_disc_input_parts.append(lr_resized)
        if self.old_hr is not None:
            real_disc_input_parts.append(self.old_hr)
        real_disc_input = torch.cat(real_disc_input_parts, dim=1)

        # Fake discriminator input
        fake_disc_input_parts = [self.output]
        if self.feed_disc_lr:
            fake_disc_input_parts.append(lr_resized)
        if self.old_hr is not None:
            fake_disc_input_parts.append(self.old_hr)
        fake_disc_input = torch.cat(fake_disc_input_parts, dim=1)

        self.optimizer_d.zero_grad()
        real_d_pred = self.net_d(real_disc_input)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        fake_d_pred = self.net_d(fake_disc_input.detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

