import importlib
import os
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.append(os.getcwd())

from ddpm import GaussianDiffusion, Trainer, Unet3DMoE
from pvqvae.pvqvae_model import PVQVAE_diff
from unet.unet import UNet as Adaptor

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _build_results_folder(cfg: DictConfig, true_bs: int) -> str:
    base = os.path.join(
        cfg.model.results_folder,
        cfg.model.results_folder_postfix,
        f"LR{cfg.model.train_lr}__BS{true_bs}__TS{cfg.model.timesteps}",
    )
    if not cfg.model.use_mask_cond:
        return base
    plane_suffix = "".join(
        f"__{p}" for p in ("use_oneplane", "use_triplane", "use_multiplane", "use_broken")
        if cfg.model.get(p, False)
    )
    return f"{base}__COND{cfg.model.cond_num}{plane_suffix}"

def _build_wandb_run_name(cfg: DictConfig, true_bs: int) -> str:
    name = (
        f"{cfg.model.results_folder_postfix}"
        f"__BS{true_bs}"
        f"__LR{cfg.model.train_lr}"
        f"__TS{cfg.model.timesteps}"
    )
    if cfg.model.use_mask_cond:
        plane_suffix = "".join(
            f"__{p}" for p in ("use_oneplane", "use_triplane", "use_multiplane", "use_broken")
            if cfg.model.get(p, False)
        )
        name = f"{name}{plane_suffix}"
    return name

@hydra.main(config_path="config", config_name="msn_cfg", version_base=None)
def run(cfg: DictConfig) -> None:
    set_seed(1)
    torch.cuda.set_device(cfg.model.gpus)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    true_bs = cfg.dataset.batch_size * cfg.model.gradient_accumulate_every
    with open_dict(cfg):
        cfg.model.results_folder = _build_results_folder(cfg, true_bs)

    os.makedirs(cfg.model.results_folder, exist_ok=True)
    with open(os.path.join(cfg.model.results_folder, "config.yaml"), "w") as fp:
        fp.write(OmegaConf.to_yaml(cfg))

    use_wandb = cfg.model.use_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.model.get("wandb_project", "GenMed"),
            name=_build_wandb_run_name(cfg, true_bs),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=cfg.model.results_folder,
            resume="allow",
        )

    if cfg.model.denoising_fn != "Unet3DMoE":
        raise ValueError(f"Unknown denoising_fn: {cfg.model.denoising_fn}")
    model = Unet3DMoE(
        dim=cfg.model.diffusion_dim,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
        out_dim=cfg.model.out_dim,
        cond_dim=cfg.dataset.cond_dim,
        use_bert_text_cond=cfg.model.use_bert_text_cond,
        use_mask_cond=cfg.model.use_mask_cond,
        cond_num=cfg.model.cond_num,
    ).cuda()

    pvqvae = PVQVAE_diff(cfg.vae.ddconfig, cfg.vae.n_embed, cfg.vae.embed_dim)
    pvqvae.load_state_dict(torch.load(cfg.vae.pvqvae_ckpt, map_location="cpu")["model"])
    pvqvae.cuda()

    adaptor_out = Adaptor(
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128],
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        pvqvae,
        adaptor_out,
        image_size=cfg.dataset.diffusion_img_size,
        num_frames=cfg.dataset.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        use_mask_guide=cfg.model.use_mask_guide,
    ).cuda()

    dataset = importlib.import_module(f"dataset.{cfg.dataset.name}_dataloader")
    train_dataloader = dataset.get_loader(cfg, augment=False)
    test_dataloader = dataset.get_loader(cfg, mode="test", augment=False)

    if cfg.dataset.name == "msn_random":
        if cfg.model.use_mask_guide:
            raise ValueError("Mask guide is not supported with the random-prompt loader.")
        if not cfg.model.use_mask_cond:
            raise ValueError("Mask conditioning is required for the random-prompt loader.")
        if cfg.model.use_oneplane:
            raise ValueError("One-plane prompts are not supported with the random-prompt loader.")
        if not (cfg.model.use_triplane and cfg.model.use_multiplane and cfg.model.use_broken):
            raise ValueError(
                "Random-prompt training expects triplane, multiplane and broken to be enabled together."
            )

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataloader,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        results_folder=cfg.model.results_folder,
        use_text_cond=cfg.model.use_bert_text_cond,
        use_mask_cond=cfg.model.use_mask_cond,
        use_mask_guide=cfg.model.use_mask_guide,
        use_wandb=use_wandb,
        test_dataset=test_dataloader,
        num_test_vis=cfg.model.get("num_test_vis", 4),
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location="cpu", strict=False)

    trainer.train()

if __name__ == "__main__":
    run()
