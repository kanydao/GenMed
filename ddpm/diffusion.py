"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import copy
import random
import time
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
import numpy as np

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from pvqvae.distributions import DiagonalGaussianDistribution

def get_gaussian_kernel1d(kernel_size: int, sigma: float):
    center = (kernel_size - 1) / 2.
    xs = torch.arange(kernel_size) - center
    kernel_1d = torch.exp(-0.5 * (xs / sigma)**2)
    return kernel_1d / kernel_1d.sum()

def get_gaussian_kernel3d(kernel_size: int, sigma: float, channels: int):
    k = get_gaussian_kernel1d(kernel_size, sigma)
    kernel_3d = k[:, None, None] * k[None, :, None] * k[None, None, :]
    kernel_3d = kernel_3d / kernel_3d.sum()
    kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    return kernel_3d.repeat(channels, 1, 1, 1, 1)

def loss_fn_triplane(img, mask):
    pos_mask = mask
    neg_mask = torch.zeros_like(mask)
    neg_mask[:, 32, :, :] = 1 - mask[:, 32, :, :]
    neg_mask[:, :, 32, :] = 1 - mask[:, :, 32, :]
    neg_mask[:, :, :, 32] = 1 - mask[:, :, :, 32]
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) + torch.sum(F.relu(-img) * neg_mask.unsqueeze(1)))\
        / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def loss_fn_oneplane(img, mask):
    pos_mask = mask
    neg_mask = torch.zeros_like(mask)
    neg_mask[:, 32, :, :] = 1 - mask[:, 32, :, :]
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) + torch.sum(F.relu(-img) * neg_mask.unsqueeze(1)))\
        / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def loss_fn_pointcloud(img, mask):
    pos_mask = torch.where(mask == 1, torch.zeros_like(mask), mask)
    neg_mask = torch.where(mask == -1, torch.zeros_like(mask), mask)
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) + torch.sum(F.relu(-img) * neg_mask.unsqueeze(1)))\
        / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def loss_fn_broken(img, mask):
    loss = F.mse_loss(img, mask.unsqueeze(1))

    return loss

def loss_fn_multiplane(img, mask):
    step = 8
    pos_mask = mask
    neg_mask = torch.zeros_like(mask)
    for i in range(0, 64, step):
        neg_mask[:, i, :, :] = 1 - mask[:, i, :, :]
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) + torch.sum(F.relu(-img) * neg_mask.unsqueeze(1)))\
        / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

def augment_sdf_with_padding(x, pad_size=96, target_size=64):
    """Augment an SDF tensor by binarizing, padding, resizing, then recomputing SDF.

    Args:
        x: Tensor [B, 1, D, H, W], SDF values approximately in [-0.2, 0.2].
        pad_size: spatial size after padding (default 96).
        target_size: output spatial size (default 64).

    Returns:
        Tensor [B, 1, D, H, W], augmented SDF.
    """
    from scipy.ndimage import distance_transform_edt

    B, C, D, H, W = x.shape
    device = x.device

    voxel = (x <= 0.02).float()

    pad_each = (pad_size - D) // 2
    pad_tuple = (pad_each, pad_each) * 3
    voxel_padded = F.pad(voxel, pad_tuple, mode='constant', value=0)

    voxel_resized = F.interpolate(
        voxel_padded,
        size=(target_size, target_size, target_size),
        mode='trilinear',
        align_corners=False
    )
    voxel_resized = (voxel_resized >= 0.5).float()

    result = torch.zeros_like(x)
    for b in range(B):
        vox_np = voxel_resized[b, 0].cpu().numpy()
        inside_dist = distance_transform_edt(vox_np)
        outside_dist = distance_transform_edt(1 - vox_np)
        sdf = (outside_dist - inside_dist) / 32.0
        sdf = sdf.clip(-0.2, 0.2)
        result[b, 0] = torch.from_numpy(sdf).float().to(device)

    return result

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        pvqvae,
        adaptor_out,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
        use_mask_guide=False,
        prompt_weight={}
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.pvqvae = pvqvae
        self.use_mask_guide = use_mask_guide

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas *\
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.text_use_bert_cls = text_use_bert_cls

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile
        self.embedding_min = -20.0
        self.embedding_max = 20.0

        self.adapt_out = nn.Sequential(
            adaptor_out,
            torch.nn.Tanh(),
        ) if exists(adaptor_out) else None

        self.prompt_weight = prompt_weight

    def prompt_loss(self, img, prompt_dicts):

        loss = 0

        for prompt_type in prompt_dicts:
            if prompt_type == 'triplane':
                loss += self.prompt_weight[prompt_type] * loss_fn_triplane(img, prompt_dicts[prompt_type].cuda())
            elif prompt_type == 'oneplane' :
                loss += self.prompt_weight[prompt_type] * loss_fn_oneplane(img, prompt_dicts[prompt_type].cuda())
            elif prompt_type == 'pointcloud' :
                loss += self.prompt_weight[prompt_type] * loss_fn_pointcloud(img, prompt_dicts[prompt_type].cuda())
            elif prompt_type == 'broken' :
                loss += self.prompt_weight[prompt_type] * loss_fn_broken(img, prompt_dicts[prompt_type].cuda())
            elif prompt_type == 'multiplane' :
                loss += self.prompt_weight[prompt_type] * loss_fn_multiplane(img, prompt_dicts[prompt_type].cuda())

        return loss

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample_one_step(self, x_prev, t):
        beta_t = extract(self.betas, t, x_prev.shape)
        alpha_t = 1.0 - beta_t
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_beta_t = torch.sqrt(beta_t)
        noise = torch.randn_like(x_prev)
        x_t = sqrt_alpha_t * x_prev + sqrt_beta_t * noise
        return x_t

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1., mask=None, return_x_start=False):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale, mask=mask))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)

        if return_x_start:
            return model_mean, posterior_variance, posterior_log_variance, x_recon

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True, mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale, mask=mask)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., xt=None, proc=True, mask=None):
        device = self.betas.device

        b = shape[0]
        if exists(xt):
            img = xt
        else:
            img = torch.randn(shape, device=device)

        it = reversed(range(0, self.num_timesteps))
        if proc:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)

        with torch.no_grad():
            for i in it:
                img = self.p_sample(img, torch.full(
                    (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale, mask=mask)

        return img

    def p_sample_loop_with_guidance_recurrent_naive(self, shape, cond=None, cond_scale=1., xt=None, proc=True, prompt_dicts=None):
        self.pvqvae.eval()
        self.denoise_fn.eval()
        for param in self.denoise_fn.parameters():
            param.requires_grad = False

        assert prompt_dicts is not None, 'mask is required for guidance'
        device = self.betas.device

        b = shape[0]
        if exists(xt):
            img = xt
        else:
            img = torch.randn(shape, device=device)

        it = reversed(range(0, self.num_timesteps))
        if proc:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)

        mask = None
        N = 1
        R = 3
        DELTA = 1.0
        for i in it:
            if self.use_mask_guide and i < 10:
                for r in range(R):
                    img_with_grad = img_guide.clone().detach().requires_grad_(True)
                    loss = 0
                    for _ in range(N):
                        img_sampled = img_with_grad
                        for k in reversed(range(0, i+1)):
                            img_sampled = self.p_sample(img_sampled, torch.full(
                                (b,), k, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale, mask=mask)

                        if self.embedding_max is not None and self.embedding_min is not None:
                            img_normalized = (((img_sampled + 1.0) / 2.0) *
                                            (self.embedding_max - self.embedding_min)) + self.embedding_min
                        else:
                            img_normalized = img_sampled
                        img_dec = self.pvqvae.decode(img_normalized)
                        loss += self.prompt_loss(img_dec, prompt_dicts)

                    loss /= N
                    loss.backward()
                    grad_norm = img_with_grad.grad.flatten(start_dim=1).norm(
                        dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    grad = img_with_grad.grad / (grad_norm + 1e-8)
                    img_guide = img_guide - grad * DELTA

                    print('t: ', i, 'mean_loss: ', loss.item(), 'max_grad: ',
                          img_with_grad.grad.abs().max().item())
                    img_with_grad.grad.zero_()
                    self.zero_grad()

                    if i > 0 and r < R-1:
                        img_guide = self.p_sample(img_guide, torch.full(
                            (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale, mask=mask)
                        img_guide = self.q_sample_one_step(img_guide, torch.full(
                            (b,), i, device=device, dtype=torch.long))
                    else:
                        break
            else:
                img_guide = img.clone()

            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale, mask=mask)
            img_guide = self.p_sample(img_guide, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale, mask=mask)

        return img_guide, img

    def p_sample_loop_with_guidance_recurrent_fast(self, shape, cond=None, cond_scale=1., xt=None, proc=True, prompt_dicts=None):
        assert prompt_dicts is not None, 'mask is required for guidance'
        device = self.betas.device
        b = shape[0]
        img = xt if exists(xt) else torch.randn(shape, device=device)
        it = reversed(range(0, self.num_timesteps))
        if proc:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        mask = prompt_dicts['triplane']
        MONTECARLO = 1
        RECURRENT = 3
        DELTA = 0.7
        p = 3
        img_guide = img.clone()
        for i in it:
            if self.use_mask_guide and i < 10:

                s = p if i >= p else i
                if i >= p:
                    with torch.no_grad():
                        x_s_det = img_guide.clone()
                        for k in range(i, s, -1):
                            t_val = torch.full((b,), k, device=device, dtype=torch.long)
                            x_s_det = self.p_sample(x_s_det, t_val.clone(), cond=cond, cond_scale=cond_scale, mask=mask)
                else:
                    x_s_det = img_guide.clone()

                x_branch = x_s_det.detach().clone().requires_grad_(True)
                loss = 0
                for _ in range(MONTECARLO):
                    x_grad = x_branch
                    for k in reversed(range(0, s+1)):
                        t_val = torch.full((b,), k, device=device, dtype=torch.long)
                        x_grad = self.p_sample(x_grad, t_val.clone(), cond=cond, cond_scale=cond_scale, mask=mask)
                    if self.embedding_max is not None and self.embedding_min is not None:
                        x_grad_norm = (((x_grad + 1.0) / 2.0) * (self.embedding_max - self.embedding_min)) + self.embedding_min
                    else:
                        x_grad_norm = x_grad
                    img_dec = self.pvqvae.decode(x_grad_norm)
                    loss += self.prompt_loss(img_dec, prompt_dicts)
                loss /= MONTECARLO
                print('t: ', i, 'mean_loss: ', loss.item())
                loss.backward()
                grad = x_branch.grad
                grad_norm = grad.flatten(1).norm(dim=1).view(b, 1, 1, 1, 1)
                updated_s = x_branch - DELTA * (grad / (grad_norm + 1e-8))

                with torch.no_grad():
                    x_forward = updated_s.detach()
                    for k in range(s, i):
                        t_val = torch.full((b,), k+1, device=device, dtype=torch.long)
                        x_forward = self.q_sample_one_step(x_forward, t_val)
                img_guide = x_forward
                x_branch.grad.zero_()
                for param in self.pvqvae.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                for param in self.denoise_fn.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                for _ in range(RECURRENT):
                    with torch.no_grad():
                        t_val = torch.full((b,), i, device=device, dtype=torch.long)
                        img_guide = self.p_sample(img_guide, t_val.clone(), cond=cond, cond_scale=cond_scale, mask=mask)
                        img_guide = self.q_sample_one_step(img_guide, t_val.clone())
            else:
                img_guide = img.clone()
            with torch.no_grad():
                t_val = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t_val.clone(), cond=cond, cond_scale=cond_scale, mask=mask)
                img_guide = self.p_sample(img_guide, t_val.clone(), cond=cond, cond_scale=cond_scale, mask=mask)
        return img_guide, img

    def p_sample_loop_with_universal_guidance(self, shape, cond=None, cond_scale=1., xt=None, proc=True, prompt_dicts=None):
        """
        Implementation of Universal Guidance with direct gradient computation on img_guide (x_t).
        This approach computes gradients directly on the current latent state and supports recurrent strategy.
        """
        assert prompt_dicts is not None, 'prompt_dicts is required for guidance'
        device = self.betas.device
        b = shape[0]
        img = xt if exists(xt) else torch.randn(shape, device=device)

        guidance_scale = 1.0
        num_guidance_steps = 1
        recurrent_steps = 3
        guidance_steps = 10

        it = reversed(range(0, self.num_timesteps))
        if proc:
            it = tqdm(it, desc='Universal Guidance sampling', total=self.num_timesteps)

        img_guide = img.clone()

        for i in it:
            t = torch.full((b,), i, device=device, dtype=torch.long)

            if self.use_mask_guide and i % 5 == 0:

                for recurrent_step in range(recurrent_steps):

                    for guidance_step in range(num_guidance_steps):

                        img_with_grad = img_guide.clone().detach().requires_grad_(True)

                        x0_estimated = self.predict_start_from_noise(
                            img_with_grad, t, noise=self.denoise_fn.forward_with_cond_scale(
                                img_with_grad, t, cond=cond, cond_scale=cond_scale))

                        if self.embedding_max is not None and self.embedding_min is not None:
                            img_normalized = (((x0_estimated + 1.0) / 2.0) *
                                            (self.embedding_max - self.embedding_min)) + self.embedding_min
                        else:
                            img_normalized = x0_estimated

                        with torch.enable_grad():
                            img_decoded = self.pvqvae.decode(img_normalized)
                            loss = self.prompt_loss(img_decoded, prompt_dicts)

                            loss.backward()
                            img_grad = img_with_grad.grad

                            grad_norm = img_grad.flatten(1).norm(dim=1).view(b, *((1,) * (len(img_grad.shape) - 1)))
                            normalized_img_grad = img_grad / (grad_norm + 1e-8)

                        img_guide = img_guide - guidance_scale * normalized_img_grad

                        img_with_grad.grad = None
                        del img_with_grad

                        if hasattr(self, 'pvqvae') and self.pvqvae is not None:
                            self.pvqvae.zero_grad()
                            for param in self.pvqvae.parameters():
                                if param.grad is not None:
                                    param.grad = None

                        if hasattr(self, 'denoise_fn') and self.denoise_fn is not None:
                            self.denoise_fn.zero_grad()
                            for param in self.denoise_fn.parameters():
                                if param.grad is not None:
                                    param.grad = None

                        if hasattr(self, 'adapt_out') and self.adapt_out is not None:
                            self.adapt_out.zero_grad()
                            for param in self.adapt_out.parameters():
                                if param.grad is not None:
                                    param.grad = None

                        torch.cuda.empty_cache()

                        if guidance_step == 0 and recurrent_step == 0:
                            print(f't: {i}, guidance_loss: {loss.item():.6f}, grad_norm: {grad_norm.mean().item():.6f}')

                    if recurrent_step < recurrent_steps - 1 and i > 0:
                        with torch.no_grad():

                            img_guide = self.p_sample(img_guide, t, cond=cond, cond_scale=cond_scale)

                            img_guide = self.q_sample_one_step(img_guide, t)

                    self.zero_grad()
                    if hasattr(self, 'pvqvae') and self.pvqvae is not None:
                        self.pvqvae.zero_grad()
                    if hasattr(self, 'denoise_fn') and self.denoise_fn is not None:
                        self.denoise_fn.zero_grad()
                    if hasattr(self, 'adapt_out') and self.adapt_out is not None:
                        self.adapt_out.zero_grad()

                    torch.cuda.empty_cache()

                with torch.no_grad():
                    img_guide = self.p_sample(img_guide, t, cond=cond, cond_scale=cond_scale)

            else:

                img_guide = self.p_sample(img_guide, t, cond=cond, cond_scale=cond_scale)

            img = self.p_sample(img, t, cond=cond, cond_scale=cond_scale)

            self.zero_grad()
            if hasattr(self, 'pvqvae') and self.pvqvae is not None:
                self.pvqvae.zero_grad()
            if hasattr(self, 'denoise_fn') and self.denoise_fn is not None:
                self.denoise_fn.zero_grad()
            if hasattr(self, 'adapt_out') and self.adapt_out is not None:
                self.adapt_out.zero_grad()

        return img_guide, img

    def p_sample_ddim(self, x, t, t_next, cond=None, cond_scale=1., clip_denoised=True, eta=0.0, mask=None):
        """
        Single DDIM sampling step
        """
        b, *_, device = *x.shape, x.device

        model_output = self.denoise_fn.forward_with_cond_scale(
            x, t, cond=cond, cond_scale=cond_scale, mask=mask)

        x_start = self.predict_start_from_noise(x, t=t, noise=model_output)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_start, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_start.ndim - 1)))

            x_start = x_start.clamp(-s, s) / s

        alpha_cumprod_t = extract(self.alphas_cumprod, t, x.shape)

        if (t_next < 0).any():

            alpha_cumprod_t_next = torch.ones_like(alpha_cumprod_t)
        else:
            alpha_cumprod_t_next = extract(self.alphas_cumprod, t_next, x.shape)

        if eta > 0:

            variance_noise = (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_next)

            variance_noise = variance_noise.clamp(min=0)
            sigma_t = eta * torch.sqrt(variance_noise)
        else:

            sigma_t = torch.zeros_like(alpha_cumprod_t)

        sqrt_alpha_next = torch.sqrt(alpha_cumprod_t_next)

        sigma_t_squared = sigma_t ** 2
        noise_coeff_arg = (1 - alpha_cumprod_t_next - sigma_t_squared).clamp(min=0)
        sqrt_one_minus_alpha_next_minus_sigma_sq = torch.sqrt(noise_coeff_arg)

        pred_mean = sqrt_alpha_next * x_start + sqrt_one_minus_alpha_next_minus_sigma_sq * model_output

        if eta > 0:
            noise = torch.randn_like(x)

            nonzero_mask = (t_next >= 0).float().view(b, *((1,) * (len(x.shape) - 1)))
            pred_sample = pred_mean + nonzero_mask * sigma_t * noise
        else:
            pred_sample = pred_mean

        return pred_sample

    def p_sample_loop_ddim(self, shape, cond=None, cond_scale=1., xt=None, timesteps=50, eta=0.0, proc=True, mask=None):
        """
        DDIM sampling loop with accelerated sampling

        Args:
            shape: output shape [B, C, F, H, W]
            timesteps: number of sampling steps (can be much fewer than training steps)
            eta: stochasticity parameter (0 for deterministic, 1 for DDPM-like)
        """
        device = self.betas.device
        b = shape[0]
        img = xt if exists(xt) else torch.randn(shape, device=device)

        times = torch.linspace(self.num_timesteps - 1, -1, timesteps + 1, device=device)
        times = torch.floor(times).long()

        time_pairs = list(zip(times[:-1], times[1:]))

        if proc:
            iterator = tqdm(time_pairs, desc=f'DDIM Sampling ({timesteps} steps, η={eta})')
        else:
            iterator = time_pairs

        with torch.no_grad():
            for t_current, t_next in iterator:

                if t_current < 0:
                    break

                t_current_batch = torch.full((b,), t_current, device=device, dtype=torch.long)
                t_next_batch = torch.full((b,), t_next, device=device, dtype=torch.long)

                img = self.p_sample_ddim(
                    img, t_current_batch, t_next_batch,
                    cond=cond, cond_scale=cond_scale, eta=eta, mask=mask
                )

        return img

    def p_sample_loop_with_universal_guidance_ddim(self, shape, cond=None, cond_scale=1., xt=None, proc=True,
                                                prompt_dicts=None, timesteps=50, eta=0.0, mask=None):
        """
        DDIM sampling with Universal Guidance for accelerated generation
        """
        assert prompt_dicts is not None, 'prompt_dicts is required for guidance'
        device = self.betas.device
        b = shape[0]
        img = xt if exists(xt) else torch.randn(shape, device=device)

        times = torch.linspace(self.num_timesteps - 1, -1, timesteps + 1, device=device)
        times = torch.floor(times).long()
        time_pairs = list(zip(times[:-1], times[1:]))

        guidance_scale = 1.0
        num_guidance_steps = 1
        recurrent_steps = 2

        guidance_frequency = max(1, len(times) // (self.num_timesteps // 5))
        guidance_timesteps = set(times[::guidance_frequency].tolist())

        if proc:
            iterator = tqdm(time_pairs, desc=f'DDIM Universal Guidance ({timesteps} steps, η={eta})')
        else:
            iterator = time_pairs

        img_guide = img.clone()

        for idx, (t_current, t_next) in enumerate(iterator):

            if t_current < 0:
                break

            t_current_batch = torch.full((b,), t_current, device=device, dtype=torch.long)
            t_next_batch = torch.full((b,), t_next, device=device, dtype=torch.long)

            if self.use_mask_guide and t_current.item() in guidance_timesteps:

                for recurrent_step in range(recurrent_steps):

                    for guidance_step in range(num_guidance_steps):

                        img_with_grad = img_guide.clone().detach().requires_grad_(True)

                        model_output = self.denoise_fn.forward_with_cond_scale(
                            img_with_grad, t_current_batch, cond=cond, cond_scale=cond_scale, mask=mask)
                        x0_estimated = self.predict_start_from_noise(
                            img_with_grad, t=t_current_batch, noise=model_output)

                        if self.embedding_max is not None and self.embedding_min is not None:
                            img_normalized = (((x0_estimated + 1.0) / 2.0) *
                                            (self.embedding_max - self.embedding_min)) + self.embedding_min
                        else:
                            img_normalized = x0_estimated

                        with torch.enable_grad():
                            img_decoded = self.pvqvae.decode(img_normalized)
                            loss = self.prompt_loss(img_decoded, prompt_dicts)

                            loss.backward()
                            img_grad = img_with_grad.grad

                            grad_norm = img_grad.flatten(1).norm(dim=1).view(b, *((1,) * (len(img_grad.shape) - 1)))
                            normalized_img_grad = img_grad / (grad_norm + 1e-8)

                        img_guide = img_guide - guidance_scale * normalized_img_grad

                        img_with_grad.grad = None
                        del img_with_grad

                        if hasattr(self, 'pvqvae') and self.pvqvae is not None:
                            self.pvqvae.zero_grad()
                            for param in self.pvqvae.parameters():
                                if param.grad is not None:
                                    param.grad = None

                        if hasattr(self, 'denoise_fn') and self.denoise_fn is not None:
                            self.denoise_fn.zero_grad()
                            for param in self.denoise_fn.parameters():
                                if param.grad is not None:
                                    param.grad = None

                        if hasattr(self, 'adapt_out') and self.adapt_out is not None:
                            self.adapt_out.zero_grad()
                            for param in self.adapt_out.parameters():
                                if param.grad is not None:
                                    param.grad = None

                        torch.cuda.empty_cache()

                        if guidance_step == 0 and recurrent_step == 0:
                            print(f't: {t_current.item()}, guidance_loss: {loss.item():.6f}, grad_norm: {grad_norm.mean().item():.6f}')

                    if recurrent_step < recurrent_steps - 1 and t_next >= 0:
                        print(f'Recurrent step {recurrent_step + 1}/{recurrent_steps} at t={t_current.item()}')
                        with torch.no_grad():

                            img_guide_denoised = self.p_sample_ddim(
                                img_guide, t_current_batch, t_next_batch,
                                cond=cond, cond_scale=cond_scale, eta=eta, mask=mask
                            )

                            if t_current > 0:

                                alpha_current = extract(self.alphas_cumprod, t_current_batch, img_guide.shape)
                                alpha_next = extract(self.alphas_cumprod, t_next_batch.clamp(min=0), img_guide.shape)

                                model_output_reverse = self.denoise_fn.forward_with_cond_scale(
                                    img_guide_denoised, t_next_batch.clamp(min=0), cond=cond, cond_scale=cond_scale, mask=mask
                                )
                                x0_from_next = self.predict_start_from_noise(
                                    img_guide_denoised, t=t_next_batch.clamp(min=0), noise=model_output_reverse
                                )

                                sqrt_alpha_current = torch.sqrt(alpha_current)
                                sqrt_one_minus_alpha_current = torch.sqrt(1 - alpha_current)

                                img_guide = sqrt_alpha_current * x0_from_next + sqrt_one_minus_alpha_current * model_output_reverse
                            else:
                                img_guide = img_guide_denoised

                    self.zero_grad()
                    if hasattr(self, 'pvqvae') and self.pvqvae is not None:
                        self.pvqvae.zero_grad()
                    if hasattr(self, 'denoise_fn') and self.denoise_fn is not None:
                        self.denoise_fn.zero_grad()
                    if hasattr(self, 'adapt_out') and self.adapt_out is not None:
                        self.adapt_out.zero_grad()

                    torch.cuda.empty_cache()

                with torch.no_grad():
                    img_guide = self.p_sample_ddim(
                        img_guide, t_current_batch, t_next_batch,
                        cond=cond, cond_scale=cond_scale, eta=eta, mask=mask
                    )
            else:

                img_guide = self.p_sample_ddim(
                    img_guide, t_current_batch, t_next_batch,
                    cond=cond, cond_scale=cond_scale, eta=eta, mask=mask
                )

            img = self.p_sample_ddim(
                img, t_current_batch, t_next_batch,
                cond=cond, cond_scale=cond_scale, eta=eta, mask=mask
            )

            self.zero_grad()
            if hasattr(self, 'pvqvae') and self.pvqvae is not None:
                self.pvqvae.zero_grad()
            if hasattr(self, 'denoise_fn') and self.denoise_fn is not None:
                self.denoise_fn.zero_grad()
            if hasattr(self, 'adapt_out') and self.adapt_out is not None:
                self.adapt_out.zero_grad()

        return img_guide, img

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = x_noisy + torch.randn_like(x_noisy) * 0.01

        if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)
        elif isinstance(cond, torch.Tensor):

            cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def sample(self, cond=None, cond_scale=1., batch_size=16, xt=None, proc=True, mask=None,
               use_ddim=False, timesteps=50, eta=0.0, adapt=True):
        """
        Main sampling method with DDIM support
        """

        if cond is None:

            txt = [
                'This is the structure of human liver.',
                'This is the structure of human left kidney.',
                'This is the structure of human stomach.',
            ]
            cond = random.choices(txt, k=batch_size)

        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        elif isinstance(cond, torch.Tensor):
            cond = cond.to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames

        if use_ddim:
            _sample = self.p_sample_loop_ddim(
                (batch_size, channels, num_frames, image_size, image_size),
                cond=cond, cond_scale=cond_scale, xt=xt, proc=proc, mask=mask,
                timesteps=timesteps, eta=eta)
        else:
            _sample = self.p_sample_loop(
                (batch_size, channels, num_frames, image_size, image_size),
                cond=cond, cond_scale=cond_scale, xt=xt, proc=proc, mask=mask)

        if self.pvqvae is not None:
            if self.embedding_max is not None and self.embedding_min is not None:
                _sample = (((_sample + 1.0) / 2.0) * (self.embedding_max -
                        self.embedding_min)) + self.embedding_min
            _sample = self.pvqvae.decode(_sample)
            _sample = _sample + self.adapt_out(_sample) * 0.2 if adapt else _sample
        _sample = _sample.clamp(-0.2, 0.2)

        return _sample

    def sample_unified(self, cond=None, cond_scale=1., batch_size=16, xt=None, proc=True, mask=None,
                      use_ddim=False, timesteps=50, eta=0.0, use_guidance=False, prompt_dicts=None):
        """
        Unified sampling interface with full control over sampling method and acceleration

        Args:
            cond: Text conditioning
            cond_scale: Classifier-free guidance scale
            batch_size: Batch size for generation
            xt: Initial noise (if None, random noise is used)
            proc: Show progress bar
            mask: Optional mask for conditional generation
            use_ddim: Whether to use DDIM (True) or DDPM (False)
            timesteps: Number of sampling steps for DDIM (ignored if use_ddim=False)
            eta: DDIM stochasticity (0=deterministic, 1=DDPM-like)
            use_guidance: Whether to use universal guidance
            prompt_dicts: Guidance prompts (required if use_guidance=True)

        Returns:
            If use_guidance=True: (guided_sample, unguided_sample)
            If use_guidance=False: sample
        """
        if use_guidance:
            return self.sample_with_guidance(
                cond=cond, cond_scale=cond_scale, batch_size=batch_size,
                xt=xt, proc=proc, prompt_dicts=prompt_dicts,
                use_ddim=use_ddim, timesteps=timesteps, eta=eta, mask=mask
            )
        else:
            return self.sample(
                cond=cond, cond_scale=cond_scale, batch_size=batch_size,
                xt=xt, proc=proc, mask=mask,
                use_ddim=use_ddim, timesteps=timesteps, eta=eta
            )

    @torch.inference_mode()
    def sample_latent_without_adaptor(self, cond=None, cond_scale=1., batch_size=16, xt=None, proc=True):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        elif isinstance(cond, torch.Tensor):
            cond = cond.to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale, xt=xt, proc=proc)

        if self.embedding_max is not None and self.embedding_min is not None:
            _sample = (((_sample + 1.0) / 2.0) * (self.embedding_max -
                    self.embedding_min)) + self.embedding_min

        return _sample

    @torch.inference_mode()
    def sample_with_adaptor(self, cond=None, cond_scale=1., batch_size=16, xt=None, proc=True, mask=None, adapt=True):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        elif isinstance(cond, torch.Tensor):
            cond = cond.to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale, xt=xt, proc=proc, mask=mask)

        if self.pvqvae is not None:
            if self.embedding_max is not None and self.embedding_min is not None:
                _sample = (((_sample + 1.0) / 2.0) * (self.embedding_max -
                        self.embedding_min)) + self.embedding_min
            _sample = self.pvqvae.decode(_sample)
            vinilla = _sample.clone()
            _sample = self.adapt_out(_sample) * 0.2 + _sample if adapt else _sample
        else:

            vinilla = _sample.clone()
        return _sample, vinilla

    def sample_with_guidance(self, cond=None, cond_scale=1., batch_size=16, xt=None, proc=True, prompt_dicts=None,
                           use_ddim=False, timesteps=50, eta=0.0, mask=None, adapt=True):
        """
        Sample with guidance support, with optional DDIM acceleration
        """
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        elif isinstance(cond, torch.Tensor):
            cond = cond.to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        begin = time.time()

        if use_ddim:
            guided_sample, _sample = self.p_sample_loop_with_universal_guidance_ddim(
                (batch_size, channels, num_frames, image_size, image_size),
                cond=cond, cond_scale=cond_scale, xt=xt, proc=proc,
                prompt_dicts=prompt_dicts, timesteps=timesteps, eta=eta, mask=mask)
        else:

            guided_sample, _sample = self.p_sample_loop_with_universal_guidance(
                (batch_size, channels, num_frames, image_size, image_size),
                cond=cond, cond_scale=cond_scale, xt=xt, proc=proc, prompt_dicts=prompt_dicts)

        print('time: ', time.time() - begin)

        if self.pvqvae is not None:
            if self.embedding_max is not None and self.embedding_min is not None:
                _sample = (((_sample + 1.0) / 2.0) * (self.embedding_max -
                        self.embedding_min)) + self.embedding_min
            _sample = self.pvqvae.decode(_sample)
            _sample = _sample + self.adapt_out(_sample) * 0.2 if adapt else _sample
            if self.embedding_max is not None and self.embedding_min is not None:
                guided_sample = (((guided_sample + 1.0) / 2.0) *
                                (self.embedding_max - self.embedding_min)) + self.embedding_min
            guided_sample = self.pvqvae.decode(guided_sample)
            guided_sample = guided_sample + self.adapt_out(guided_sample) * 0.2 if adapt else guided_sample
        else:
            _sample = _sample * 0.2
            guided_sample = guided_sample * 0.2
        _sample = _sample.clamp(-0.2, 0.2)
        guided_sample = guided_sample.clamp(-0.2, 0.2)
        return guided_sample, _sample

    def inference(self, x, cond=None, mask=None, use_ddim=False, timesteps=50, eta=0.0, adapt=None):
        """
        Regular inference with optional DDIM acceleration

        Args:
            x: Input tensor
            cond: Text conditioning
            mask: Optional mask for conditional generation
            use_ddim: Whether to use DDIM (True) or DDPM (False)
            timesteps: Number of DDIM sampling steps (ignored if use_ddim=False)
            eta: DDIM stochasticity (0=deterministic, 1=DDPM-like)
        """
        if self.pvqvae is not None:
            self.pvqvae.eval()
        self.denoise_fn.eval()

        with torch.no_grad():
            if self.pvqvae is not None:
                posterior = self.pvqvae.encode_whole_fold(x)
                x = posterior.mode() if isinstance(
                    posterior, DiagonalGaussianDistribution) else posterior
            b, device, img_size, = x.shape[0], x.device, self.image_size
            check_shape(x, 'b c f h w', c=self.channels,
                        f=self.num_frames, h=img_size, w=img_size)

            xt = None

            if use_ddim:

                img = self.sample(
                    batch_size=b, xt=xt, proc=False, cond=cond, mask=mask,
                    use_ddim=True, timesteps=timesteps, eta=eta, adapt=adapt)

                if self.pvqvae is not None:

                    vinilla = img.clone()
                else:
                    vinilla = img.clone()
            else:

                img, vinilla = self.sample_with_adaptor(
                    batch_size=b, xt=xt, proc=False, cond=cond, mask=mask, adapt=adapt)

            if self.pvqvae is not None:

                img0 = self.pvqvae.decode(x+torch.randn_like(x))
                kernel = get_gaussian_kernel3d(3, 1.0, 1).cuda()
                img1 = self.pvqvae.decode(x+torch.randn_like(x))
                img1 = F.conv3d(img1, weight=kernel, padding=1, groups=1)
            else:
                img0 = vinilla
                img1 = vinilla
            print(img.min().item(), img.max().item())
        return img, vinilla, img0, img1

    def inference_latent(self, x):
        with torch.no_grad():
            poseterior = self.pvqvae.encode_whole_fold(x)
            x = poseterior.mode() if isinstance(
                poseterior, DiagonalGaussianDistribution) else poseterior
            b, device, img_size, = x.shape[0], x.device, self.image_size
            check_shape(x, 'b c f h w', c=self.channels,
                        f=self.num_frames, h=img_size, w=img_size)

            xt = self.q_sample(x, torch.full((b,), self.num_timesteps - 1, device=device))
            x0 = self.sample_latent_without_adaptor(batch_size=b, xt=xt, proc=False)
        return x0

    def inference_with_guidance(self, x, cond=None, prompt_dicts=None, use_ddim=False, timesteps=20, eta=0.0, adapt=True):
        """
        Inference with guidance support and optional DDIM acceleration
        """
        self.pvqvae.eval()
        self.denoise_fn.eval()

        with torch.no_grad():
            posterior = self.pvqvae.encode_whole_fold(x)
            x = posterior.mode() if isinstance(
                posterior, DiagonalGaussianDistribution) else posterior
            b, device, img_size, = x.shape[0], x.device, self.image_size
            check_shape(x, 'b c f h w', c=self.channels,
                        f=self.num_frames, h=img_size, w=img_size)

        xt = None
        img, vinilla = self.sample_with_guidance(
            batch_size=b, xt=xt, proc=False, cond=cond, prompt_dicts=prompt_dicts,
            use_ddim=use_ddim, timesteps=timesteps, eta=eta, adapt=adapt)
        with torch.no_grad():
            img0 = self.pvqvae.decode(x)
            img1 = self.pvqvae.decode(x+torch.randn_like(x))

        return img, vinilla, img0, img1

    def forward(self, x, *args, **kwargs):

        orig = x.detach().clone()

        if self.pvqvae is not None:
            with torch.no_grad():
                posterior = self.pvqvae.encode_whole_fold(x)
            x = posterior.sample() if isinstance(posterior, DiagonalGaussianDistribution) else posterior

            xn = (x + torch.randn_like(x)).detach()
            if self.embedding_max is not None and self.embedding_min is not None:
                x = ((x - self.embedding_min) / (self.embedding_max - self.embedding_min)) * 2.0 - 1.0
        else:
            x = x * 5.0

        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels,
                    f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        diff_loss = self.p_losses(x, t, *args, **kwargs)

        self._last_diff_loss = diff_loss.item()
        self._last_adapt_loss = 0.0

        if self.pvqvae is not None:
            with torch.no_grad():

                imgn = self.pvqvae.decode(xn) + torch.randn_like(orig) * 0.01
            imgdn = self.adapt_out(imgn) * 0.2 + imgn
            if self.loss_type == 'l1':
                adapt_loss = F.l1_loss(orig, imgdn)
            elif self.loss_type == 'l2':
                adapt_loss = F.mse_loss(orig, imgdn)
            elif self.loss_type == 'huber':
                adapt_loss = F.huber_loss(orig, imgdn)
            adapt_loss = adapt_loss * 20.0
            self._last_adapt_loss = adapt_loss.item()
            print(diff_loss.item(), adapt_loss.item())
        else:
            adapt_loss = 0

        loss = diff_loss + adapt_loss
        return loss

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=16,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        dataset=None,
        *,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder=None,
        max_grad_norm=None,
        use_text_cond=False,
        use_mask_cond=False,
        use_mask_guide=False,
        is_distributed=False,
        rank=0,
        test_dataset=None,
        num_test_vis=1,
        log_every_n_steps=1,
        use_tensorboard=True,
        use_wandb=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.cfg = cfg

        self.len_dataloader = len(dataset)
        self.dl = cycle(dataset)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        self.initial_lr = train_lr
        self.train_num_steps = train_num_steps

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.results_folder = Path(results_folder)

        self.use_text_cond = use_text_cond
        self.use_mask_cond = use_mask_cond
        self.use_mask_guide = use_mask_guide

        self.is_distributed = is_distributed
        self.rank = rank

        self.test_dataset = test_dataset
        self.num_test_vis = num_test_vis
        self.test_vis_folder = self.results_folder / 'test_visualizations'

        self.log_every_n_steps = log_every_n_steps
        self.use_tensorboard = use_tensorboard and (not self.is_distributed or self.rank == 0)
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.results_folder / 'tensorboard'))
                print(f'TensorBoard logging enabled: {self.results_folder / "tensorboard"}')
                print(f'  View with: tensorboard --logdir {self.results_folder / "tensorboard"}')
            except ImportError:
                print('Warning: tensorboard not available. Install with: pip install tensorboard')
                self.use_tensorboard = False

        self.use_wandb = use_wandb

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):

        if self.is_distributed and self.rank != 0:
            return

        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        data['model'] = {k.replace('module.', ''): v for k, v in data['model'].items()}
        self.model.load_state_dict(data['model'], **kwargs)
        data['ema'] = {k.replace('module.', ''): v for k, v in data['ema'].items()}
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def visualize_test_samples(self, milestone):
        """
        Visualize test samples during training by generating reconstructions
        and creating side-by-side 3D mesh GIFs.

        Args:
            milestone: Current training milestone for naming output files
        """

        if self.is_distributed and self.rank != 0:
            return

        if self.test_dataset is None:
            return

        print(f'\n[Milestone {milestone}] Generating test visualizations...')

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        from utils.plot_3d import create_side_by_side_mesh_gif
        import numpy as np

        try:
            from utils.metrics import compute_metrics_cond
            use_pytorch3d_metrics = True
            print("  pytorch3d metrics available and enabled")
        except ImportError:
            use_pytorch3d_metrics = False
            print("  Note: pytorch3d not available. Only Dice score will be computed.")

        self.test_vis_folder.mkdir(exist_ok=True, parents=True)
        milestone_folder = self.test_vis_folder / f'milestone_{milestone}'
        milestone_folder.mkdir(exist_ok=True, parents=True)

        self.ema_model.eval()
        if hasattr(self.ema_model, 'pvqvae') and self.ema_model.pvqvae is not None:
            self.ema_model.pvqvae.eval()
        if hasattr(self.ema_model, 'denoise_fn'):
            self.ema_model.denoise_fn.eval()
        if hasattr(self.ema_model, 'adapt_out'):
            self.ema_model.adapt_out.eval()

        metrics_list = []

        with torch.no_grad():

            collected_images = []
            collected_conds  = []
            collected_organs = []
            cond_is_tensor   = None

            for batch_data in self.test_dataset:
                if len(collected_images) >= self.num_test_vis:
                    break

                batch_images = batch_data['image']
                batch_cond   = batch_data['text'] if self.use_text_cond else None
                batch_organ  = batch_data.get('organ_type', ['unknown'] * batch_images.shape[0])

                collected_images.append(batch_images[0:1])
                organ_type = batch_organ[0] if isinstance(batch_organ, list) else batch_organ[0]
                collected_organs.append(organ_type)

                if batch_cond is None:
                    collected_conds.append(None)
                elif isinstance(batch_cond, torch.Tensor):
                    collected_conds.append(batch_cond[0:1])
                    cond_is_tensor = True
                elif isinstance(batch_cond, list):
                    collected_conds.append(batch_cond[0])
                    cond_is_tensor = False
                else:
                    collected_conds.append(batch_cond)

            n_samples = len(collected_images)
            print(f'  Collected {n_samples} categories, running batched inference...')

            batch_image = torch.cat(collected_images, dim=0).cuda()

            if collected_conds[0] is None:
                batch_cond_in = None
            elif cond_is_tensor:
                batch_cond_in = torch.cat(collected_conds, dim=0).cuda()
            else:
                batch_cond_in = collected_conds

            try:
                recon_batch, vanilla_batch, dec_batch, noise_batch = self.ema_model.inference(
                    batch_image, cond=batch_cond_in, use_ddim=False, timesteps=10, eta=0.0
                )
            except RuntimeError as e:
                print(f"  Batched inference failed ({e}), falling back to single-sample mode...")
                recon_batch = vanilla_batch = dec_batch = noise_batch = None

            per_sample = []
            valid_indices = []

            for i, organ_type in enumerate(collected_organs):
                origin_img = collected_images[i].cuda()

                if recon_batch is not None:
                    recon   = recon_batch[i:i+1]
                    vanilla = vanilla_batch[i:i+1]
                else:

                    if cond_is_tensor:
                        cond_i = collected_conds[i].cuda()
                    elif cond_is_tensor is False:
                        cond_i = [collected_conds[i]]
                    else:
                        cond_i = None
                    try:
                        recon, vanilla, _, _ = self.ema_model.inference(
                            origin_img, cond=cond_i, use_ddim=False, timesteps=10, eta=0.0
                        )
                    except RuntimeError as e2:
                        print(f"    Skipping sample {i}: {e2}")
                        continue

                per_sample.append({
                    'idx':       i,
                    'organ':     organ_type,
                    'origin_t':  origin_img,
                    'recon_t':   recon,
                    'vanilla_t': vanilla,
                })
                valid_indices.append(len(per_sample) - 1)

            all_origin_t  = torch.cat([s['origin_t'][:,0]  for s in per_sample], dim=0)
            all_recon_t   = torch.cat([s['recon_t'][:,0]   for s in per_sample], dim=0)
            all_vanilla_t = torch.cat([s['vanilla_t'][:,0] for s in per_sample], dim=0)

            origin_mask_b  = (all_origin_t  < 0).float()
            recon_mask_b   = (all_recon_t   < 0).float()
            vanilla_mask_b = (all_vanilla_t < 0).float()

            spatial = (1, 2, 3)
            inter_r = (origin_mask_b * recon_mask_b).sum(dim=spatial)
            inter_v = (origin_mask_b * vanilla_mask_b).sum(dim=spatial)
            sum_o   = origin_mask_b.sum(dim=spatial)
            sum_r   = recon_mask_b.sum(dim=spatial)
            sum_v   = vanilla_mask_b.sum(dim=spatial)

            recon_dice_batch   = (2 * inter_r / (sum_o + sum_r   + 1e-8)).cpu().numpy()
            vanilla_dice_batch = (2 * inter_v / (sum_o + sum_v   + 1e-8)).cpu().numpy()
            del all_origin_t, all_recon_t, all_vanilla_t
            del origin_mask_b, recon_mask_b, vanilla_mask_b

            cd_per_sample  = [0.0] * len(per_sample)
            uhd_per_sample = [0.0] * len(per_sample)

            if use_pytorch3d_metrics and len(per_sample) > 0:
                try:
                    from utils.metrics import voxel_to_pointcloud, batch_uhd_manual
                    from pytorch3d.loss import chamfer_distance as cd_fn

                    all_recon  = torch.cat([s['recon_t'][:,0]  for s in per_sample], dim=0)
                    all_origin = torch.cat([s['origin_t'][:,0] for s in per_sample], dim=0)
                    resolution = all_recon.shape[-1]
                    print(f'  Running batched point-cloud extraction on {all_recon.shape[0]} samples...')
                    gen_pc  = voxel_to_pointcloud(all_recon)  / float(resolution)
                    real_pc = voxel_to_pointcloud(all_origin) / float(resolution)

                    cd_vals  = cd_fn(gen_pc, real_pc, batch_reduction=None)[0]

                    uhd_vals = batch_uhd_manual(gen_pc, real_pc)
                    for k in range(len(per_sample)):
                        cd_per_sample[k]  = cd_vals[k].item()
                        uhd_per_sample[k] = uhd_vals[k].item()
                    print(f'  Batch CD={cd_vals.mean().item():.4f}, UHD={uhd_vals.mean().item():.4f}')
                    del all_recon, all_origin, gen_pc, real_pc, cd_vals, uhd_vals
                except Exception as e:
                    print(f'  Warning: batched CD/UHD failed ({e}), values set to 0.')

            for k, s in enumerate(per_sample):
                idx        = s['idx']
                organ_type = s['organ']
                origin_np  = s['origin_t'][0, 0].cpu().numpy()
                recon_np   = s['recon_t'][0, 0].detach().cpu().numpy()
                print(f'\nPost-processing sample {idx} ({organ_type})...')

                recon_dice   = float(recon_dice_batch[k])
                vanilla_dice = float(vanilla_dice_batch[k])
                cd_val  = cd_per_sample[k]
                uhd_val = uhd_per_sample[k]

                metrics_list.append({
                    'idx': idx, 'organ_type': organ_type,
                    'recon_dice': recon_dice, 'vanilla_dice': vanilla_dice,
                    'cd': cd_val, 'uhd': uhd_val,
                })

                if use_pytorch3d_metrics:
                    print(f'Sample {idx} ({organ_type}): Dice={recon_dice:.4f}, CD={cd_val:.4f}, UHD={uhd_val:.4f}')
                else:
                    print(f'Sample {idx} ({organ_type}): Dice={recon_dice:.4f}')

                organ_folder = milestone_folder / organ_type.replace(' ', '_')
                organ_folder.mkdir(exist_ok=True, parents=True)
                create_side_by_side_mesh_gif(
                    origin_np, recon_np,
                    str(organ_folder), f'sample_{idx:04d}',
                    threshold=0.02,
                    title=f'{organ_type} - Milestone {milestone}'
                )

            del batch_image, recon_batch, vanilla_batch, dec_batch, noise_batch, per_sample
            gc.collect()
            torch.cuda.empty_cache()

        import json

        cd_values = [m['cd'] for m in metrics_list if m['cd'] > 0]
        uhd_values = [m['uhd'] for m in metrics_list if m['uhd'] > 0]

        metrics_summary = {
            'milestone': milestone,
            'step': self.step,
            'samples': metrics_list,
            'mean_recon_dice': float(np.mean([m['recon_dice'] for m in metrics_list])),
            'mean_vanilla_dice': float(np.mean([m['vanilla_dice'] for m in metrics_list])),
            'mean_cd': float(np.mean(cd_values)) if cd_values else 0.0,
            'mean_uhd': float(np.mean(uhd_values)) if uhd_values else 0.0,
            'pytorch3d_enabled': use_pytorch3d_metrics,
        }

        with open(milestone_folder / 'metrics.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        print(f'\nMean Recon Dice: {metrics_summary["mean_recon_dice"]:.4f}')
        if use_pytorch3d_metrics:
            print(f'Mean CD: {metrics_summary["mean_cd"]:.4f}')
            print(f'Mean UHD: {metrics_summary["mean_uhd"]:.4f}')
        else:
            print(f'(CD/UHD metrics disabled - pytorch3d not in use)')
        print(f'Test visualizations saved to: {milestone_folder}\n')

        self.model.train()

    def train(
        self,
        null_cond_prob=0.,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        self.results_folder.mkdir(exist_ok=True, parents=True)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                image = data['image'].cuda()
                text = data['text'] if self.use_text_cond else None

                mask = None
                if self.use_mask_cond :
                    prompt_dicts = data.get('prompt_dicts')
                    tensors = list(prompt_dicts.values())
                    mask = torch.stack(tensors, dim=1)
                mask = mask.cuda() if exists(mask) else None

                if self.step == 0 and i == 0:
                    print("Mask shape:", mask.shape if mask is not None else "None")

                with autocast(enabled=self.amp):
                    loss = self.model(
                        image,
                        cond=text,
                        mask=mask,
                        null_cond_prob=null_cond_prob,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                if i == self.gradient_accumulate_every - 1 and self.step % 20 == 0:
                    if not self.is_distributed or (self.is_distributed and self.rank == 0):
                        print(f'{self.step}: {loss.item():.4f}, lr: {self.opt.param_groups[0]["lr"]}, cfg: {str(self.results_folder).split("/")[-1]}')

            log = {'loss': loss.item()}

            grad_norm = None
            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)

                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.use_tensorboard and self.step % self.log_every_n_steps == 0:

                self.writer.add_scalar('Loss/total', loss.item(), self.step)
                self.writer.add_scalar('Loss/diffusion', self.model._last_diff_loss, self.step)
                self.writer.add_scalar('Loss/adaptor', self.model._last_adapt_loss, self.step)

                self.writer.add_scalar('Training/learning_rate', self.opt.param_groups[0]['lr'], self.step)

                if grad_norm is not None:
                    self.writer.add_scalar('Training/gradient_norm', grad_norm, self.step)

                self.writer.add_scalar('Training/step', self.step, self.step)

                if self.step % (self.log_every_n_steps * 100) == 0:
                    self.writer.flush()

            if self.use_wandb and self.step % self.log_every_n_steps == 0:
                try:
                    import wandb
                    wandb_log = {
                        'loss/total': loss.item(),
                        'train/learning_rate': self.opt.param_groups[0]['lr'],
                        'train/step': self.step,
                    }

                    if hasattr(self.model, '_last_diff_loss'):
                        wandb_log['loss/diffusion'] = self.model._last_diff_loss
                    if hasattr(self.model, '_last_adapt_loss'):
                        wandb_log['loss/adaptor'] = self.model._last_adapt_loss
                    if grad_norm is not None:
                        wandb_log['train/gradient_norm'] = grad_norm
                    wandb.log(wandb_log, step=self.step)
                except Exception as e:
                    print(f'Warning: wandb logging failed at step {self.step}: {e}')

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                if self.use_mask_cond==False:
                    self.visualize_test_samples(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')

        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
            print('TensorBoard logging closed')

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
                print('wandb run finished')
            except Exception:
                pass
