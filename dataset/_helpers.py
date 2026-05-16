"""Shared helpers for the MSN data loaders.

Provides:
- Prompt generators: triplane, oneplane, broken, multiplane.
- A dispatcher `get_prompt(sdf, prompt_type)`.
- `augment_sdf` for affine 3D augmentation.
- `data_collate` for batching SDFs, text embeddings, and prompt dicts.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage
import torch

from utils.sdf_fn import compute_sdf

VOLUME_SIZE = 64
SLICE_INDEX = VOLUME_SIZE // 2

def break_voxel(voxel: np.ndarray, num_areas: int = 5) -> np.ndarray:
    """Randomly carve / fill spherical regions in a binary voxel grid."""
    for _ in range(num_areas):
        if np.random.rand() > 0.5:
            indices = np.argwhere(voxel == 1)
            fill_value = 0
        else:
            indices = np.argwhere(voxel == 0)
            fill_value = 1
        if len(indices) == 0:
            continue
        center = indices[np.random.choice(len(indices))]
        radius = np.random.uniform(1, 5)
        x, y, z = np.indices(voxel.shape)
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        voxel[distance <= radius] = fill_value
    return voxel

def get_triplane_prompt(sdf: np.ndarray) -> torch.Tensor:
    mask = sdf < 0
    plane = np.zeros_like(sdf)
    plane[SLICE_INDEX, :, :] = mask[SLICE_INDEX, :, :]
    plane[:, SLICE_INDEX, :] = mask[:, SLICE_INDEX, :]
    plane[:, :, SLICE_INDEX] = mask[:, :, SLICE_INDEX]
    return torch.from_numpy(plane).unsqueeze(0).long()

def get_oneplane_prompt(sdf: np.ndarray) -> torch.Tensor:
    mask = sdf < 0
    plane = np.zeros_like(sdf)
    plane[SLICE_INDEX, :, :] = mask[SLICE_INDEX, :, :]
    return torch.from_numpy(plane).unsqueeze(0).long()

def get_broken_prompt(sdf: np.ndarray) -> torch.Tensor:
    mask = break_voxel(sdf < 0)
    mask = compute_sdf(mask)
    return torch.from_numpy(mask).unsqueeze(0).float()

def get_multiplane_prompt(sdf: np.ndarray, step: int = 8) -> torch.Tensor:
    mask = sdf < 0
    plane = np.zeros_like(sdf)
    for i in range(0, VOLUME_SIZE, step):
        plane[i, :, :] = mask[i, :, :]
    return torch.from_numpy(plane).unsqueeze(0).long()

_PROMPT_FNS = {
    "triplane": get_triplane_prompt,
    "oneplane": get_oneplane_prompt,
    "broken": get_broken_prompt,
    "multiplane": get_multiplane_prompt,
}

def get_prompt(sdf: np.ndarray, prompt_type: str) -> torch.Tensor:
    if prompt_type not in _PROMPT_FNS:
        raise NotImplementedError(f"Unknown prompt_type: {prompt_type}")
    return _PROMPT_FNS[prompt_type](sdf)

def get_prompt_random(sdf: np.ndarray, prompt_type: str) -> torch.Tensor:
    """Random-prompt variant: 50% chance to return an all-zero prompt."""
    if prompt_type == "oneplane":
        raise NotImplementedError("Random sampling is not defined for the oneplane prompt.")
    if np.random.randint(0, 2) == 0:
        return torch.zeros(1, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE, dtype=torch.long)
    return get_prompt(sdf, prompt_type)

def augment_sdf(
    sdf: np.ndarray,
    translation_range=None,
    rotation_range=None,
    scale_range=None,
    padding_value: float = 1e6,
) -> np.ndarray:
    """Affine 3D augmentation (translation, rotation, anisotropic scaling)."""
    if translation_range is None:
        translation_range = (0, 0)
    if rotation_range is None:
        rotation_range = (0, 0)
    if scale_range is None:
        scale_range = (1, 1)
    if isinstance(translation_range, (int, float)):
        translation_range = (-translation_range, translation_range)
    if isinstance(rotation_range, (int, float)):
        rotation_range = (-rotation_range, rotation_range)
    assert isinstance(scale_range, tuple)

    def _expand(rng):
        if len(rng) == 1:
            return (-rng[0], rng[0]) * 3
        if len(rng) == 2:
            return tuple(rng) * 3
        assert len(rng) == 6
        return tuple(rng)

    tr = _expand(translation_range)
    ro = _expand(rotation_range)
    sc = _expand(scale_range) if len(scale_range) != 2 else tuple(scale_range) * 3

    translation = tuple(np.random.uniform(tr[2 * i], tr[2 * i + 1]) for i in range(3))
    rotation = tuple(np.random.uniform(ro[2 * i], ro[2 * i + 1]) for i in range(3))
    scale = tuple(np.random.uniform(sc[2 * i], sc[2 * i + 1]) for i in range(3))

    scale_matrix = np.diag([1 / scale[0], 1 / scale[1], 1 / scale[2], 1])
    rx, ry, rz = np.deg2rad(rotation)
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    affine = scale_matrix @ (Rx @ Ry @ Rz)

    augmented = scipy.ndimage.affine_transform(
        sdf,
        matrix=affine[:3, :3],
        offset=np.array(translation),
        order=3,
        mode="constant",
        cval=padding_value,
    )
    return augmented * np.linalg.norm(scale) / np.linalg.norm([1, 1, 1])

def data_collate(batch):
    img = torch.cat([item[0] for item in batch])
    text_embeddings = torch.stack([item[1] for item in batch])
    categories = [item[2] for item in batch]
    prompt_dicts: dict = {}
    for prompt_dict in (item[3] for item in batch):
        for key, value in prompt_dict.items():
            prompt_dicts.setdefault(key, []).append(value)
    for key in prompt_dicts:
        prompt_dicts[key] = torch.cat(prompt_dicts[key], dim=0)
    return {
        "image": img,
        "text": text_embeddings,
        "prompt_dicts": prompt_dicts,
        "organ_type": categories,
    }
