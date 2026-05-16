"""Adapted from https://github.com/lucidrains/video-diffusion-pytorch.

Adds a conditional MoE block on top of the base Unet3D.
"""

import math
import copy
import os
import random
import time
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

def loss_fn_triplane(img, mask):
    pos_mask = mask
    neg_mask = torch.zeros_like(mask)
    neg_mask[:, 32, :, :] = 1 - mask[:, 32, :, :]
    neg_mask[:, :, 32, :] = 1 - mask[:, :, 32, :]
    neg_mask[:, :, :, 32] = 1 - mask[:, :, :, 32]
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) +
            torch.sum(F.relu(-img) * neg_mask.unsqueeze(1))) / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def loss_fn_oneplane(img, mask):
    pos_mask = mask
    neg_mask = torch.zeros_like(mask)
    neg_mask[:, 32, :, :] = 1 - mask[:, 32, :, :]
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) +
            torch.sum(F.relu(-img) * neg_mask.unsqueeze(1))) / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def loss_fn_pointcloud(img, mask):
    pos_mask = torch.where(mask == 1, torch.zeros_like(mask), mask)
    neg_mask = torch.where(mask == -1, torch.zeros_like(mask), mask)
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) +
            torch.sum(F.relu(-img) * neg_mask.unsqueeze(1))) / (torch.sum(pos_mask) + torch.sum(neg_mask))
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
    loss = (torch.sum(F.relu(img) * pos_mask.unsqueeze(1)) +
            torch.sum(F.relu(-img) * neg_mask.unsqueeze(1))) / (torch.sum(pos_mask) + torch.sum(neg_mask))
    return loss

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class TextConditionedMoE(nn.Module):
    def __init__(self, in_dim, out_dim, bert_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_experts)])
        self.router = nn.Linear(bert_dim, num_experts)

    def forward(self, x, cond):
        router_logits = self.router(cond)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_vals, topk_indices = router_probs.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(router_probs)
        mask.scatter_(1, topk_indices, topk_vals)
        mask = mask.unsqueeze(1).unsqueeze(1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(expert_outputs * mask, dim=-1)
        return output

class ResnetBlockMoE(nn.Module):
    def __init__(self, dim, dim_out, bert_dim, *, time_emb_dim=None, groups=8, num_experts=4, top_k=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = nn.Sequential(
            nn.Conv3d(dim, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.moe = TextConditionedMoE(in_dim=dim_out, out_dim=dim_out, bert_dim=bert_dim, num_experts=num_experts, top_k=top_k)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time_emb must be provided"
            ts = self.mlp(time_emb)
            ts = rearrange(ts, 'b c -> b c 1 1 1')
            scale_shift = ts.chunk(2, dim=1)
        h = self.block1(x)
        if scale_shift is not None:
            h = h * (scale_shift[0] + 1) + scale_shift[1]
        b, c, f, h_, w_ = h.shape
        h_flat = rearrange(h, 'b c f h w -> b (f h w) c')
        h_moe = self.moe(h_flat, cond)
        h_moe = rearrange(h_moe, 'b (f h w) c -> b c f h w', f=f, h=h_, w=w_)
        return h_moe + self.res_conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        if exists(pos_bias):
            sim = sim + pos_bias
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)
            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        bert_dim=768,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        use_mask_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8,
        use_moe=True,
        num_experts=4,
        top_k=2,
        cond_num=0
    ):
        super().__init__()
        self.channels = channels
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        def temporal_attn(dim):
            return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.use_mask_cond = use_mask_cond
        if use_mask_cond:
            self.mask_conv = nn.Sequential(
                nn.Conv3d(cond_num, 8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv3d(32, channels, kernel_size=3, stride=1, padding=1)
            )
            self.init_conv = nn.Sequential(
                nn.Conv3d(channels*2, init_dim, (init_kernel_size, init_kernel_size, init_kernel_size),
                          padding=(init_padding, init_padding, init_padding)),
                nn.Conv3d(init_dim, init_dim, kernel_size=1)
            )
        else:
            self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.cond_mlp = None
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        if use_bert_text_cond and not exists(cond_dim):
            cond_dim = BERT_MODEL_DIM

        if exists(cond_dim):
            bert_dim = cond_dim
        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        total_cond_dim = time_dim + int(cond_dim or 0)
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if use_moe:
                block1 = ResnetBlockMoE(dim_in, dim_out, bert_dim=bert_dim, time_emb_dim=total_cond_dim, groups=resnet_groups, num_experts=num_experts, top_k=top_k)
                block2 = ResnetBlockMoE(dim_out, dim_out, bert_dim=bert_dim, time_emb_dim=total_cond_dim, groups=resnet_groups, num_experts=num_experts, top_k=top_k)
            else:
                block1 = ResnetBlock(dim_in, dim_out, time_emb_dim=total_cond_dim, groups=resnet_groups)
                block2 = ResnetBlock(dim_out, dim_out, time_emb_dim=total_cond_dim, groups=resnet_groups)
            self.downs.append(nn.ModuleList([
                block1,
                block2,
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
        mid_dim = dims[-1]
        if use_moe:
            self.mid_block1 = ResnetBlockMoE(mid_dim, mid_dim, bert_dim=bert_dim, time_emb_dim=total_cond_dim, groups=resnet_groups, num_experts=num_experts, top_k=top_k)
            self.mid_block2 = ResnetBlockMoE(mid_dim, mid_dim, bert_dim=bert_dim, time_emb_dim=total_cond_dim, groups=resnet_groups, num_experts=num_experts, top_k=top_k)
        else:
            self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=total_cond_dim, groups=resnet_groups)
            self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=total_cond_dim, groups=resnet_groups)
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            if use_moe:
                block1 = ResnetBlockMoE(dim_out * 2, dim_in, bert_dim=bert_dim, time_emb_dim=total_cond_dim, groups=resnet_groups, num_experts=num_experts, top_k=top_k)
                block2 = ResnetBlockMoE(dim_in, dim_in, bert_dim=bert_dim, time_emb_dim=total_cond_dim, groups=resnet_groups, num_experts=num_experts, top_k=top_k)
            else:
                block1 = ResnetBlock(dim_out * 2, dim_in, time_emb_dim=total_cond_dim, groups=resnet_groups)
                block2 = ResnetBlock(dim_in, dim_in, time_emb_dim=total_cond_dim, groups=resnet_groups)
            self.ups.append(nn.ModuleList([
                block1,
                block2,
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ResnetBlock(dim * 2, dim, groups=resnet_groups),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(self, *args, cond_scale=2., **kwargs):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, cond=None, mask=None, null_cond_prob=0., focus_present_mask=None, prob_focus_present=0.):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        if self.use_mask_cond:

            mask_emb = self.mask_conv(mask.float())
            x = torch.cat((x, mask_emb), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        cond = self.cond_mlp(cond) if exists(self.cond_mlp) else cond
        if cond is not None:
            cond = cond.to(device)
        if self.has_cond:
            mask_prob = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask_prob, 'b -> b 1'), self.null_cond_emb.to(device), cond)
            t = torch.cat((t, cond), dim=-1)
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, time_emb=t, cond=cond) if hasattr(block1, 'moe') else block1(x, t)
            x = block2(x, time_emb=t, cond=cond) if hasattr(block2, 'moe') else block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t, cond=cond) if hasattr(self.mid_block1, 'moe') else self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t, cond=cond) if hasattr(self.mid_block2, 'moe') else self.mid_block2(x, t)
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, cond=cond) if hasattr(block1, 'moe') else block1(x, t)
            x = block2(x, t, cond=cond) if hasattr(block2, 'moe') else block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)
