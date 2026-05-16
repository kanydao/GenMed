"""Zero-shot MSN data loader: evaluates on held-out organ categories."""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset._helpers import data_collate, get_prompt

class MsnZeroShotDataset(Dataset):
    """Returns (sdf_tensor, text_embedding, category, prompt_dict) on zero-shot classes."""

    def __init__(self, cfg, mode: str = "test", augment: bool = False):
        super().__init__()
        if mode == "train":
            raise ValueError("Zero-shot dataset does not support training mode.")
        if mode not in ("test", "uncond"):
            raise ValueError(f"Invalid mode: {mode}")

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
        embedding_file = cfg.dataset.get("text_embedding_file", "text_embeddings.json")
        self.text_embeddings = self._load_text_embeddings(embedding_file)

        with open(os.path.join("data_preproc", "zero_shot_category.json"), "r") as f:
            self.grouped_cases = json.load(f)

        self.caselist = [
            (category, file_path)
            for category in sorted(self.grouped_cases.keys())
            for file_path in self.grouped_cases[category]
        ]
        self.num_types = len(self.grouped_cases)

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
            sdf = np.load(full_path)
        except Exception:
            sdf = np.zeros((64, 64, 64), dtype=np.float32)
        sdf = sdf.clip(-0.2, 0.2)
        sdf_tensor = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0)

        text_embedding = self.text_embeddings.get(category)
        if text_embedding is None:
            text_embedding = torch.zeros(5120, dtype=torch.float32)
        prompts = {pt: get_prompt(sdf, pt) for pt in self.prompt_type_list}
        return sdf_tensor, text_embedding, category, prompts

    def get_cases_by_type(self, case_type: str):
        return self.grouped_cases.get(case_type, [])

class _TypeBatchSampler:
    def __init__(self, dataset: MsnZeroShotDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.groups = [
            (case_type, list(range(len(dataset.grouped_cases[case_type]))))
            for case_type in sorted(dataset.grouped_cases.keys())
        ]

    def __iter__(self):
        for case_type, indices in self.groups:
            yield [
                self.dataset.caselist.index((case_type, self.dataset.grouped_cases[case_type][i]))
                for i in indices
            ]

    def __len__(self) -> int:
        return len(self.groups)

def get_loader(cfg, mode: str = "test", augment: bool = False) -> DataLoader:
    assert mode in ("test", "uncond")
    dataset = MsnZeroShotDataset(cfg, mode=mode, augment=augment)
    sampler = _TypeBatchSampler(dataset, cfg.dataset["batch_size"])
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.dataset["num_workers"],
        collate_fn=data_collate,
    )
