import copy
import gc
import importlib
import json
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ddpm import GaussianDiffusion, Trainer, Unet3DMoE
from pvqvae.pvqvae_model import PVQVAE_diff
from unet.unet import UNet as Adaptor
from utils.metrics import compute_metrics_cond
from utils.plot_3d import create_side_by_side_mesh_gif

def _build_diffusion(cfg: DictConfig) -> GaussianDiffusion:
    if cfg.model.denoising_fn != "Unet3DMoE":
        raise ValueError(f"Unknown denoising_fn: {cfg.model.denoising_fn}")
    model = Unet3DMoE(
        dim=cfg.model.diffusion_dim,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
        out_dim=cfg.model.out_dim,
        cond_dim=cfg.dataset.get("cond_dim"),
        use_bert_text_cond=cfg.model.use_bert_text_cond,
        use_mask_cond=cfg.model.use_mask_cond,
        cond_num=cfg.model.cond_num,
    ).cuda()

    pvqvae = PVQVAE_diff(cfg.vae.ddconfig, cfg.vae.n_embed, cfg.vae.embed_dim)
    adaptor_out = Adaptor(
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128],
    ).cuda()

    prompt_weight = {
        "triplane": cfg.model.triplane_weight,
        "oneplane": cfg.model.oneplane_weight,
        "pointcloud": 0,
        "broken": cfg.model.broken_weight,
        "multiplane": cfg.model.multiplane_weight,
    }
    return GaussianDiffusion(
        model,
        pvqvae,
        adaptor_out,
        image_size=cfg.dataset.diffusion_img_size,
        num_frames=cfg.dataset.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        use_mask_guide=cfg.model.use_mask_guide,
        prompt_weight=prompt_weight,
    ).cuda()

def _release(trainer: Trainer) -> None:
    trainer.ema_model.zero_grad()
    for attr in ("pvqvae", "denoise_fn", "adapt_out"):
        sub = getattr(trainer.ema_model, attr, None)
        if sub is not None:
            sub.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

@hydra.main(config_path="config", config_name="msn_cfg", version_base=None)
def run(cfg: DictConfig) -> None:
    torch.cuda.set_device(cfg.model.gpus)

    dataset_mod = importlib.import_module(f"dataset.{cfg.dataset.name}_dataloader")
    test_loader = dataset_mod.get_loader(cfg, mode="test", augment=False)

    cfg.model.results_folder = os.path.join(
        cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix
    )

    diffusion = _build_diffusion(cfg)
    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=test_loader,
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
    )
    trainer.load(cfg.model.checkpoint_for_sample, map_location="cpu", strict=True)

    if cfg.model.use_mask_cond and cfg.model.use_mask_guide:
        raise ValueError("`use_mask_cond` and `use_mask_guide` are mutually exclusive.")

    output_dir = cfg.model.output_dir
    os.makedirs(output_dir, exist_ok=True)

    minibatch_size = cfg.dataset.minibatch_size
    draw_gif = True
    use_ddim = False
    use_adapt = True

    recon_dice_list, vanilla_dice_list, cd_list, uhd_list = [], [], [], []
    dice_dict: dict = {}

    for idx, data in enumerate(tqdm(test_loader), start=1):
        image = data["image"].cuda()
        cond = data["text"] if cfg.model.use_bert_text_cond else None
        prompt_dicts = data.get("prompt_dicts")
        mask = torch.stack(list(prompt_dicts.values()), dim=1).cuda()
        organ_type = data["organ_type"][0]

        create_side_by_side_mesh_gif(
            image[0, 0].cpu().numpy(),
            mask[0, 0].cpu().numpy(),
            os.path.join(output_dir, organ_type),
            f"{idx:04d}_{organ_type}_broken",
            threshold=0.02,
            title=f"Original vs Mask - {organ_type}",
        )

        recon, vanilla, dec, noise = [], [], [], []
        for j in range(0, len(image), minibatch_size):
            image_b = copy.deepcopy(image[j:j + minibatch_size])
            cond_b = copy.deepcopy(cond[j:j + minibatch_size]) if cond is not None else None
            mask_b = copy.deepcopy(mask[j:j + minibatch_size])
            prompts_b = {k: copy.deepcopy(v[j:j + minibatch_size]) for k, v in prompt_dicts.items()}

            if cfg.model.use_mask_cond:
                r, v, d, n = trainer.ema_model.inference(
                    image_b, cond=cond_b, mask=mask_b, use_ddim=use_ddim, adapt=use_adapt
                )
            elif cfg.model.use_mask_guide:
                r, v, d, n = trainer.ema_model.inference_with_guidance(
                    image_b, cond=cond_b, prompt_dicts=prompts_b, use_ddim=use_ddim, adapt=use_adapt
                )
            else:
                r, v, d, n = trainer.ema_model.inference(
                    image_b, cond=cond_b, mask=None, use_ddim=use_ddim, adapt=use_adapt
                )

            recon.append(r.detach().cpu())
            vanilla.append(v.detach().cpu())
            dec.append(d.detach().cpu())
            noise.append(n.detach().cpu())

            del r, v, d, n, image_b, cond_b, mask_b, prompts_b
            _release(trainer)

        recon = torch.cat(recon, dim=0)
        vanilla = torch.cat(vanilla, dim=0)

        recon_dice = vanilla_dice = 0.0
        for j in range(image.shape[0]):
            origin_mask = (image[j, 0].detach().cpu().numpy() < 0).astype(int)
            recon_mask = (recon[j, 0].numpy() < 0).astype(int)
            vanilla_mask = (vanilla[j, 0].numpy() < 0).astype(int)
            recon_dice += 2 * np.sum(origin_mask * recon_mask) / (
                np.sum(origin_mask) + np.sum(recon_mask) + 1e-8
            )
            vanilla_dice += 2 * np.sum(origin_mask * vanilla_mask) / (
                np.sum(origin_mask) + np.sum(vanilla_mask) + 1e-8
            )
        recon_dice /= image.shape[0]
        vanilla_dice /= image.shape[0]
        recon_dice_list.append(recon_dice)
        vanilla_dice_list.append(vanilla_dice)

        metrics = compute_metrics_cond(recon[:, 0], image[:, 0])
        cd, uhd = metrics["cd"], metrics["uhd"]
        cd_list.append(cd)
        uhd_list.append(uhd)

        dice_dict.setdefault(organ_type, []).append((recon_dice, vanilla_dice, cd, uhd))

        if draw_gif:
            create_side_by_side_mesh_gif(
                image[0, 0].cpu().numpy(),
                recon[0, 0].cpu().numpy(),
                os.path.join(output_dir, organ_type),
                f"{idx:04d}_{organ_type}",
                threshold=0.02,
                title=f"Original vs Reconstructed - {organ_type}",
            )

        mean_metrics = {
            "recon_dice": float(np.mean(recon_dice_list)),
            "vanilla_dice": float(np.mean(vanilla_dice_list)),
            "cd": float(np.mean(cd_list)),
            "uhd": float(np.mean(uhd_list)),
        }
        with open(os.path.join(output_dir, "class_metrics.json"), "w") as f:
            json.dump(dice_dict, f)
        with open(os.path.join(output_dir, "mean_metrics.json"), "w") as f:
            json.dump(mean_metrics, f)

        del image, cond, prompt_dicts, recon, vanilla
        _release(trainer)

if __name__ == "__main__":
    run()
