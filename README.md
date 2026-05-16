# GenMed

Reference implementation accompanying the GenMed paper
([arXiv:2605.10645](https://arxiv.org/pdf/2605.10645)).

A latent diffusion model for 3D medical SDF generation with mask-prompt
classifier-free guidance over four prompt types: `triplane`, `oneplane`,
`broken`, `multiplane`.

## Repository layout

```
.
├── train.py                 # diffusion training entry point
├── sample.py                # sampling entry point
├── config/                  # Hydra configs (msn_cfg.yaml is the root)
├── ddpm/                    # diffusion process + UNet wrapper
├── unet/                    # adaptor UNet
├── pvqvae/                  # patch-based latent VAE
├── dataset/                 # MSN data loaders (standard / zero-shot / random)
└── utils/                   # SDF helpers, metrics, 3D plotting
```

## Install

```bash
pip install -r requirements.txt
```

## Data

The MSN data loaders expect:

- `${root_dir}/<class_name>/<sample_id>.npy` — 64^3 SDF volumes,
  values in `[-0.2, 0.2]`.
- `data_preproc/<train|test|zero_shot>_data_category.json` — splits, mapping
  `class_name -> [relative .npy paths]`.
- `data_preproc/text_embeddings.json` — `class_name -> List[float]` of
  length `cond_dim` (5120 in the default config).
- `pretrained_models/pvqvae.pth` — pretrained PVQVAE checkpoint.

## Train

```bash
python train.py \
    dataset.root_dir=/path/to/msn_sdf \
    vae.pvqvae_ckpt=./pretrained_models/pvqvae.pth
```

Any field in `config/` is overridable via Hydra. Enable Weights & Biases
with `model.use_wandb=True` and `export WANDB_API_KEY=...`.

## Sample

```bash
python sample.py \
    dataset.root_dir=/path/to/msn_sdf \
    vae.pvqvae_ckpt=./pretrained_models/pvqvae.pth \
    model.checkpoint_for_sample=./runs/genmed/.../model-XX.pt \
    model.output_dir=./samples
```

## License

MIT — see [`LICENSE`](LICENSE).
