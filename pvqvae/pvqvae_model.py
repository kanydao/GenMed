

from __future__ import print_function

import importlib
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.utils.data
from torch import nn
from torch.nn import init

from einops import rearrange

from pvqvae.modules import Encoder3D, Decoder3D
from pvqvae.quantizer import VectorQuantizer
from pvqvae.distributions import DiagonalGaussianDistribution

def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    for m in net.children():
        m.apply(init_func)

def _maybe_to_container(config: Any) -> Any:
    """Convert OmegaConf configs to standard Python containers when available."""
    if config is None or isinstance(config, (dict, list)):
        return config
    try:
        from omegaconf import OmegaConf

        if config.__class__.__name__ in {"DictConfig", "ListConfig"}:
            return OmegaConf.to_container(config, resolve=True)
    except Exception:

        return config
    return config

def _import_from_string(target: str) -> Any:
    module, attr = target.rsplit('.', 1)
    mod = importlib.import_module(module)
    return getattr(mod, attr)

class PVQVAE(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 remap=None,
                 sane_index_shape=False,
                 ignore_keys=[],
                 ):
        super(PVQVAE, self).__init__()

        self.ddconfig = ddconfig

        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=1.0,
                                        remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            init_weights(self.encoder, 'normal', 0.02)
            init_weights(self.decoder, 'normal', 0.02)
            init_weights(self.quant_conv, 'normal', 0.02)
            init_weights(self.post_quant_conv, 'normal', 0.02)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return quant, emb_loss, info

    def encode_whole(self, x):
        x = self.unfold_to_cubes(x)
        h = self.encoder(x)
        h = self.quant_conv(h)

        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_not_quant(self, h):
        quant, _, _ = self.quantize(h, is_voxel=True)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_from_quant(self,quant_code):
        embed_from_code = self.quantize.embedding(quant_code)
        return embed_from_code

    def decode_enc_idices(self, enc_indices, z_spatial_dim=8):

        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices)
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3', d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        dec = self.decode(z_q)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, verbose=False):
        quant, diff, info = self.encode(input)
        dec = self.decode(quant)

        if verbose:
            return dec, quant, diff, info
        else:
            return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    @staticmethod

    def unfold_to_cubes(x, cube_size=8, stride=8):
        """
            assume x.shape: b, c, d, h, w
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

class PVQVAE_diff(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 remap=None,
                 sane_index_shape=False,
                 ignore_keys=[],
                 ):
        super(PVQVAE_diff, self).__init__()

        self.ddconfig = ddconfig

        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=1.0,
                                        remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        self.quant_conv = torch.nn.Conv3d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            init_weights(self.encoder, 'normal', 0.02)
            init_weights(self.decoder, 'normal', 0.02)
            init_weights(self.quant_conv, 'normal', 0.02)
            init_weights(self.post_quant_conv, 'normal', 0.02)

    @staticmethod

    def unfold_to_cubes(x, cube_size=8, stride=8):
        """
            assume x.shape: b, c, d, h, w
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def encode_whole(self, x):
        x = self.unfold_to_cubes(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def encode_whole_fold(self, x):
        x = self.unfold_to_cubes(x)
        h = self.encoder(x)
        moments = self.fold_to_voxels_simple(self.quant_conv(h))
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_from_quant(self, quant_code):
        embed_from_code = self.quantize.embedding(quant_code)
        return embed_from_code

    def decode_enc_idices(self, enc_indices, z_spatial_dim=8):

        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices)
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3', d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        dec = self.decode(z_q)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    @staticmethod

    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size)
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                      p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    @staticmethod

    def fold_to_voxels_simple(x_cubes, ncubes_per_dim=8):
        batch_size = int(x_cubes.shape[0] / (ncubes_per_dim * ncubes_per_dim * ncubes_per_dim))
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size)
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                      p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def forward(self, input, batch_size, ncubes_per_dim, verbose=False):

        posterior = self.encode(input)
        z = posterior.sample()
        z_voxel = self.fold_to_voxels(z, batch_size=batch_size, ncubes_per_dim=ncubes_per_dim)
        dec = self.decode(z_voxel)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

class PVQVAELoss(nn.Module):
    """VAE-style reconstruction + KL loss used by legacy PVQVAE training."""

    def __init__(
        self,
        recon_loss_type: str = "l1",
        recon_weight: float = 1.0,
        kl_weight: float = 1e-6,
        perceptual_weight: float = 0.0,
        perceptual_target: Optional[str] = None,
        perceptual_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        recon_loss_type = recon_loss_type.lower()
        if recon_loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported reconstruction loss '{recon_loss_type}'.")

        self.recon_loss_type = recon_loss_type
        self.recon_weight = float(recon_weight)
        self.kl_weight = float(kl_weight)
        self.perceptual_weight = float(perceptual_weight)

        self.perceptual = None
        if self.perceptual_weight > 0.0:
            if perceptual_target is None:
                raise ValueError("perceptual_target must be provided when perceptual_weight > 0")
            params = perceptual_params or {}
            params = _maybe_to_container(params) or {}
            if not isinstance(params, dict):
                params = dict(params)
            perceptual_cls = _import_from_string(perceptual_target)
            self.perceptual = perceptual_cls(**params)
            self.perceptual.eval()
            for param in self.perceptual.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posterior: Optional[DiagonalGaussianDistribution],
        optimizer_idx: int = 0,
        global_step: int = 0,
        last_layer: Optional[torch.Tensor] = None,
        split: str = "train",
    ):
        del optimizer_idx, global_step, last_layer

        if self.recon_loss_type == "l1":
            rec_component = torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean()
        else:
            diff = inputs.contiguous() - reconstructions.contiguous()
            rec_component = torch.pow(diff, 2).mean()

        perceptual_component = inputs.new_tensor(0.0)
        if self.perceptual is not None:
            perceptual_component = self.perceptual(inputs.contiguous(), reconstructions.contiguous()).mean()

        nll_loss = self.recon_weight * rec_component + self.perceptual_weight * perceptual_component

        if posterior is None:
            kl_raw = inputs.new_tensor(0.0)
        else:
            kl_values = posterior.kl()
            kl_raw = kl_values.mean()

        kl_loss = self.kl_weight * kl_raw
        total_loss = nll_loss + kl_loss

        log = OrderedDict()
        log[f"{split}/loss"] = total_loss.detach()
        log[f"{split}/nll_loss"] = nll_loss.detach()
        log[f"{split}/rec_loss"] = rec_component.detach()
        log[f"{split}/kl_loss"] = kl_loss.detach()
        if self.perceptual is not None:
            log[f"{split}/perceptual_loss"] = perceptual_component.detach()
        log[f"{split}/kl_raw"] = kl_raw.detach()

        return total_loss, log

def build_pvqvae_loss(lossconfig: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Instantiate a PVQVAE loss module following the legacy config contract."""
    if lossconfig is None:
        return PVQVAELoss()

    cfg = _maybe_to_container(lossconfig)
    if cfg is None:
        return PVQVAELoss()

    if isinstance(cfg, dict):
        params = _maybe_to_container(cfg.get('params', {})) or {}
        if not isinstance(params, dict):
            params = dict(params)

        target = cfg.get('target')
        if target is None:
            return PVQVAELoss(**params)
        loss_cls = _import_from_string(target)
        return loss_cls(**params)

    target = getattr(cfg, 'target', None)
    params = _maybe_to_container(getattr(cfg, 'params', {})) or {}
    if not isinstance(params, dict):
        params = dict(params)

    if target is None:
        return PVQVAELoss(**params)
    loss_cls = _import_from_string(target)
    return loss_cls(**params)
