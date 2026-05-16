"""Random-prompt MSN data loader: each prompt is set to all-zero with probability 0.5.

Used to train the mask-conditional null-prompt branch (classifier-free guidance).
"""
from __future__ import annotations

import json
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset._helpers import augment_sdf, data_collate, get_prompt_random

def _resolve_data_json(mode: str) -> str:
    if mode == "train":
        return "train_data_category.json"
    if mode == "test":
        return "test_data_category.json"
    raise ValueError(f"Invalid mode: {mode}")

class MsnRandomDataset(Dataset):
    """Returns (sdf_tensor, text_embedding, category, prompt_dict) with randomized prompts."""

    def __init__(self, cfg, mode: str = "train", augment: bool = False):
        super().__init__()
        self.prompt_type_list = [
            label
            for label, flag in zip(
                ("triplane", "broken", "oneplane", "multiplane"),
                (
                    cfg.model.use_triplane,
                    cfg.model.use_broken,
                    cfg.model.use_oneplane,
                    cfg.model.use_multiplane,
                ),
            )
            if flag
        ]

        self.root_dir = cfg.dataset["root_dir"]
        self.training = mode == "train"
        self.augment = augment

        embedding_file = cfg.dataset.get("text_embedding_file", "text_embeddings.json")
        self.text_embeddings = self._load_text_embeddings(embedding_file)

        data_json = _resolve_data_json(mode)
        with open(os.path.join("data_preproc", data_json), "r") as f:
            self.grouped_cases = json.load(f)

        self.caselist = [
            (category, file_path)
            for category in sorted(self.grouped_cases.keys())
            for file_path in self.grouped_cases[category]
        ]
        self.num_types = len(self.grouped_cases)

        if self.training and self.augment:
            self.augment_sdf = partial(
                augment_sdf, translation_range=0, rotation_range=3, scale_range=(0.8, 1.0)
            )

    @staticmethod
    def _load_text_embeddings(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text embedding file not found: {path}")
        with open(path, "r") as f:
            raw = json.load(f)
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}

    def __len__(self) -> int:
        return len(self.caselist)

    def __getitem__(self, idx: int):
        category, file_path = self.caselist[idx]
        full_path = os.path.join(self.root_dir, file_path)
        try:
            sdf = np.load(full_path).astype(np.float32)
        except Exception:
            sdf = np.zeros((64, 64, 64), dtype=np.float32)
        if self.augment:
            sdf = self.augment_sdf(sdf)

        sdf_tensor = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0)
        text_embedding = self.text_embeddings.get(category)
        if text_embedding is None:
            text_embedding = torch.zeros(5120, dtype=torch.float32)
        prompts = {pt: get_prompt_random(sdf, pt) for pt in self.prompt_type_list}
        return sdf_tensor, text_embedding, category, prompts

    def get_cases_by_type(self, case_type: str):
        return self.grouped_cases.get(case_type, [])

class _TypeBatchSampler:
    def __init__(self, dataset: MsnRandomDataset, batch_size: int, training: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.training = training
        self.groups = [
            (case_type, list(range(len(dataset.grouped_cases[case_type]))))
            for case_type in sorted(dataset.grouped_cases.keys())
        ]

    def __iter__(self):
        groups = random.sample(self.groups, len(self.groups)) if self.training else self.groups
        for case_type, indices in groups:
            if self.training:
                if len(indices) < self.batch_size:
                    indices = (
                        indices * (self.batch_size // len(indices))
                        + indices[: self.batch_size % len(indices)]
                    )
                else:
                    indices = random.sample(indices, self.batch_size)
            yield [
                self.dataset.caselist.index((case_type, self.dataset.grouped_cases[case_type][i]))
                for i in indices
            ]

    def __len__(self) -> int:
        return len(self.groups)

def get_loader(cfg, mode: str = "train", augment: bool = False) -> DataLoader:
    assert mode in ("train", "test")
    dataset = MsnRandomDataset(cfg, mode=mode, augment=augment)
    sampler = _TypeBatchSampler(dataset, cfg.dataset["batch_size"], training=(mode == "train"))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.dataset["num_workers"],
        collate_fn=data_collate,
    )
