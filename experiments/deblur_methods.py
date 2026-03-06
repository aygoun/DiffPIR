"""
Deblurring method helpers for experiments.

These functions are intended to mirror the SR helpers from `sr_methods.py`,
but for the deblurring task.
"""

from __future__ import annotations
from tqdm import tqdm

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from utils import utils_image as util
from utils import utils_model
from utils import utils_sisr as sr
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from .common import ImageResult, MethodConfig


@dataclass
class DeblurPlaceholder:
    """Placeholder to keep the API surface for future extensions."""

    name: str = "deblur_placeholder"


@dataclass
class DiffPIRDeblurHyperParams:
    """
    Container for all DiffPIR deblurring hyper-parameters.

    Defaults mirror `main_ddpir_deblur.py` so that experiments are
    comparable, while remaining easy to override via `MethodConfig.extra`.
    """

    # noise and model
    noise_level_img: float = 12.75 / 255.0
    noise_level_model: Optional[float] = None
    model_name: str = "diffusion_ffhq_10m"
    num_train_timesteps: int = 1000

    # sampling loop
    iter_num: int = 100
    iter_num_U: int = 1
    skip_type: str = "quad"  # "uniform" or "quad"
    eta: float = 0.0
    zeta_default: float = 0.1

    # guidance / data-consistency
    sub_1_analytic: bool = True
    guidance_scale: float = 1.0

    # blur
    use_DIY_kernel: bool = True
    blur_mode: str = "Gaussian"  # "Gaussian" or "motion"
    kernel_size: int = 61
    kernel_std: float = 3.0

    # bookkeeping
    show_img: bool = False
    save_L: bool = True
    save_E: bool = True
    save_LEH: bool = False
    save_progressive: bool = False
    border: int = 0

    # LPIPS
    calc_LPIPS: bool = True

    sf: int = 1


def _build_hparams_from_cfg(cfg: MethodConfig) -> DiffPIRDeblurHyperParams:
    """Create hyper-parameter object, allowing overrides through `cfg.extra`."""

    extra: Dict[str, Any] = cfg.extra or {}
    hp = DiffPIRDeblurHyperParams()

    # Couple global experiment config into the hyper-params
    hp.zeta_default = float(cfg.zeta)

    # Optional user overrides (kept small on purpose)
    for key in [
        "noise_level_img",
        "noise_level_model",
        "model_name",
        "iter_num",
        "iter_num_U",
        "skip_type",
        "eta",
        "guidance_scale",
        "use_DIY_kernel",
        "blur_mode",
        "kernel_size",
        "kernel_std",
        "calc_LPIPS",
    ]:
        if key in extra:
            setattr(hp, key, extra[key])

    # If model noise not specified, keep it equal to image noise (as in main file)
    if hp.noise_level_model is None:
        hp.noise_level_model = hp.noise_level_img

    return hp


def run_diffpir_deblur(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR deblurring runner from the original code.
    """

    assert (
        cfg.task == "deblur"
    ), f"DiffPIR deblur expects task='deblur', got {cfg.task!r}"
    assert cfg.generate_mode == "DiffPIR", "run_diffpir_deblur is specific to DiffPIR."

    hp = _build_hparams_from_cfg(cfg)

    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = logging.getLogger(f"diffpir_deblur.{img_name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info(
        "Starting DiffPIR deblurring | img=%s | lambda=%.3f | zeta=%.3f | noise=%.4f | model=%s",
        img_name,
        cfg.lambda_,
        cfg.zeta,
        hp.noise_level_img,
        hp.model_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    beta_start = 0.1 / 1000
    beta_end = 20 / 1000
    betas = np.linspace(beta_start, beta_end, hp.num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(
        sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
    )  # equivalent noise sigma on image

    sigma = max(0.001, hp.noise_level_img)

    noise_inti_img = 50 / 255
    t_start = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img)
    t_start = hp.num_train_timesteps - 1

    # Model loading
    cwd = ""
    model_zoo = os.path.join(cwd, "model_zoo")
    model_path = os.path.join(model_zoo, hp.model_name + ".pt")

    model_config = (
        dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        )
        if hp.model_name == "diffusion_ffhq_10m"
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # Optional LPIPS network
    loss_fn_vgg = None
    if hp.calc_LPIPS:
        import lpips

        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    # ------------------------------------------------------------------
    # Prepare single image, blur kernel and noisy observation
    # ------------------------------------------------------------------
    n_channels = 3

    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)
    img_H = util.modcrop(img_H, 8)

    # kernel: match the DIY-kernel branch of the original implementation
    if hp.use_DIY_kernel:
        np.random.seed(seed=0)
        if hp.blur_mode == "Gaussian":
            kernel_std_i = hp.kernel_std * np.abs(np.random.rand() * 2 + 1)
            kernel = GaussialBlurOperator(
                kernel_size=hp.kernel_size, intensity=kernel_std_i, device=device
            )
        elif hp.blur_mode == "motion":
            kernel = MotionBlurOperator(
                kernel_size=hp.kernel_size, intensity=hp.kernel_std, device=device
            )
        else:
            raise ValueError(f"Unknown blur_mode {hp.blur_mode!r}")
        k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
        k = k_tensor.clone().detach().cpu().numpy()
        k = np.squeeze(k)
    else:
        import hdf5storage

        kernels = hdf5storage.loadmat(os.path.join(cwd, "kernels", "Levin09.mat"))[
            "kernels"
        ]
        k = kernels[0, 0].astype(np.float32)

    # blur the GT image to create the low-quality observation
    logger.info(
        "Applying %s blur (kernel_size=%d, kernel_std≈%.3f) and AWGN (σ≈%.4f)",
        hp.blur_mode,
        hp.kernel_size,
        hp.kernel_std,
        hp.noise_level_img,
    )
    img_L = ndimage.convolve(img_H, np.expand_dims(k, axis=2), mode="wrap")
    if hp.show_img:
        util.imshow(img_L)
    img_L = util.uint2single(img_L)

    np.random.seed(seed=0)
    img_L = img_L * 2 - 1
    img_L += np.random.normal(0, hp.noise_level_img * 2, img_L.shape)
    img_L = img_L / 2 + 0.5

    # Build tensors
    y = util.single2tensor4(img_L).to(device)  # (1, 3, H, W)

    # For y with given noise level, add noise from t_y
    t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * hp.noise_level_img)
    sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * (2 * y - 1) + torch.sqrt(
        sqrt_1m_alphas_cumprod[t_start] ** 2
        - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y] ** 2
    ) * torch.randn_like(y)

    # For deblurring we work at the native resolution (no SR scaling),
    # so the effective scale factor used in the FFT pre-computation is 1.
    hp.sf = 1
    k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, hp.sf)

    # ------------------------------------------------------------------
    # Pre-compute sigmas / rhos for the whole trajectory
    # ------------------------------------------------------------------
    sigmas = []
    sigma_ks = []
    rhos = []
    for i in range(hp.num_train_timesteps):
        sigmas.append(reduced_alpha_cumprod[hp.num_train_timesteps - 1 - i])
        if cfg.generate_mode == "DiffPIR":
            sigma_ks.append(sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i])
        else:
            sigma_ks.append(torch.sqrt(betas[i] / alphas[i]))
        rhos.append(cfg.lambda_ * (sigma**2) / (sigma_ks[i] ** 2))
    rhos = torch.tensor(rhos).to(device)
    sigmas = torch.tensor(sigmas).to(device)
    sigma_ks = torch.tensor(sigma_ks).to(device)

    # ------------------------------------------------------------------
    # Build the time-step sequence
    # ------------------------------------------------------------------
    skip = hp.num_train_timesteps // hp.iter_num
    if hp.skip_type == "uniform":
        seq = [i * skip for i in range(hp.iter_num)]
        if skip > 1:
            seq.append(hp.num_train_timesteps - 1)
    elif hp.skip_type == "quad":
        seq = np.sqrt(np.linspace(0, hp.num_train_timesteps**2, hp.iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
    else:
        raise ValueError(f"Unknown skip_type {hp.skip_type!r}")

    # ------------------------------------------------------------------
    # Main reverse diffusion loop (single image)
    # ------------------------------------------------------------------
    logger.info(
        "Starting reverse diffusion (%d steps, skip_type=%s)", len(seq), hp.skip_type
    )
    x_0 = None
    for i in tqdm(range(len(seq)), desc="DiffPIR Deblurring"):
        curr_sigma = sigmas[seq[i]].cpu().numpy()
        t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
        if t_i > t_start:
            continue

        for u in range(hp.iter_num_U):
            # 1) Reverse diffusion step
            x0 = utils_model.model_fn(
                x,
                noise_level=curr_sigma * 255,
                model_out_type="pred_xstart",
                model_diffusion=model,
                diffusion=diffusion,
                ddim_sample=False,
                alphas_cumprod=alphas_cumprod,
            )

            # 2) Data-consistency step (analytic solution branch from original code)
            if seq[i] != seq[-1] and cfg.generate_mode == "DiffPIR":
                if hp.sub_1_analytic:
                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                    if i < hp.num_train_timesteps:
                        x0_p = x0 / 2 + 0.5
                        x0_p = sr.data_solution(
                            x0_p.float(), FB, FBC, F2B, FBFy, tau, hp.sf
                        )
                        x0_p = x0_p * 2 - 1
                        x0 = x0 + hp.guidance_scale * (x0_p - x0)

            # 3) Step to next time index if not at final step
            if cfg.generate_mode == "DiffPIR" and not (
                seq[i] == seq[-1] and u == hp.iter_num_U - 1
            ):
                t_im1 = utils_model.find_nearest(
                    reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy()
                )
                eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                eta_sigma = (
                    cfg.zeta
                    * sqrt_1m_alphas_cumprod[t_im1]
                    / sqrt_1m_alphas_cumprod[t_i]
                    * torch.sqrt(betas[t_i])
                )
                x = (
                    sqrt_alphas_cumprod[t_im1] * x0
                    + np.sqrt(1 - cfg.zeta)
                    * (
                        torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma**2)
                        * eps
                        + eta_sigma * torch.randn_like(x)
                    )
                    + np.sqrt(cfg.zeta)
                    * sqrt_1m_alphas_cumprod[t_im1]
                    * torch.randn_like(x)
                )

            if u < hp.iter_num_U - 1 and seq[i] != seq[-1]:
                t_im1 = utils_model.find_nearest(
                    reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy()
                )
                sqrt_alpha_effective = (
                    sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                )
                x = sqrt_alpha_effective * x + torch.sqrt(
                    sqrt_1m_alphas_cumprod[t_i] ** 2
                    - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1] ** 2
                ) * torch.randn_like(x)

        x_0 = x / 2 + 0.5

    assert x_0 is not None
    logger.info("Reverse diffusion finished for %s", img_name)

    # ------------------------------------------------------------------
    # Metrics + saving outputs
    # ------------------------------------------------------------------
    img_E = util.tensor2uint(x_0)

    psnr = util.calculate_psnr(img_E, img_H, border=hp.border)
    lpips_score: Optional[float] = None
    if loss_fn_vgg is not None:
        img_H_tensor = np.transpose(img_H, (2, 0, 1))
        img_H_tensor = torch.from_numpy(img_H_tensor)[None, :, :, :].to(device)
        img_H_tensor = img_H_tensor / 255 * 2 - 1
        lpips_tensor = loss_fn_vgg(x_0.detach() * 2 - 1, img_H_tensor)
        lpips_score = float(lpips_tensor.cpu().detach().numpy()[0][0][0][0])

    logger.info(
        "Finished DiffPIR deblurring | img=%s | PSNR=%.3f dB | LPIPS=%s",
        img_name,
        psnr,
        f"{lpips_score:.4f}" if lpips_score is not None else "N/A",
    )

    # Output folder (portable, per-method)
    extra = cfg.extra or {}
    root_out = extra.get("output_root", "outputs")
    method_out = os.path.join(root_out, "diffpir_deblur")
    util.mkdir(method_out)

    # Save restored image (and optionally LR)
    out_est = os.path.join(method_out, f"{img_name}_diffpir{ext}")
    util.imsave(img_E, out_est)
    if hp.save_L:
        out_lr = os.path.join(method_out, f"{img_name}_LR{ext}")
        util.imsave(util.single2uint(img_L), out_lr)
        logger.info("Saved LR image to %s", out_lr)

    logger.info("Saved deblurred image to %s", out_est)

    return ImageResult(psnr=float(psnr), lpips=lpips_score)


def run_dps_deblur(
    img_path: str, cfg: MethodConfig, mode: str = "DPS_y0"
) -> ImageResult:
    """
    Single-image DPS deblurring runner (to be implemented by you).

    `mode` should typically be either \"DPS_y0\" or \"DPS_yt\" and should
    control how the data-consistency term is applied in the loop.
    """

    raise NotImplementedError("run_dps_deblur is not implemented yet.")


def run_pnp_drunet_deblur(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play deblurring runner using DRUNet or another `Denoiser` prior.

    Recommended design:
    - Construct a DRUNetDenoiser (or other denoiser) instance.
    - Implement an iterative PnP scheme:
      * data step using the blur operator
      * prior step using the denoiser
    - Return an `ImageResult` with metrics.
    """

    raise NotImplementedError("run_pnp_drunet_deblur is not implemented yet.")
