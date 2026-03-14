"""
Inpainting method helpers for experiments.

These functions mirror the structure of `deblur_methods.py` but for the
inpainting task, adapting the logic from `main_ddpir_inpainting.py`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from guided_diffusion.script_util import (
    NUM_CLASSES,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils import utils_image as util
from utils import utils_model
from utils.utils_inpaint import mask_generator

from .common import ImageResult, MethodConfig
from .pnp_priors import GaussianDenoiser, DRUNetDenoiser, DiffBIRDenoiser, Denoiser


# ===========================================================================
# Hyper-parameter dataclasses
# ===========================================================================


@dataclass
class DiffPIRInpaintHyperParams:
    """
    Container for all DiffPIR inpainting hyper-parameters.

    Defaults mirror `main_ddpir_inpainting.py` so that experiments are
    comparable, while remaining easy to override via `MethodConfig.extra`.
    """

    # noise and model
    noise_level_img: float = 0.0
    noise_level_model: Optional[float] = None
    model_name: str = "diffusion_ffhq_10m"
    num_train_timesteps: int = 1000

    # sampling loop
    iter_num: int = 20
    iter_num_U: int = 1
    skip_type: str = "quad"  # "uniform" or "quad"
    eta: float = 0.0
    zeta_default: float = 1.0

    # guidance / data-consistency
    sub_1_analytic: bool = True
    guidance_scale: float = 1.0

    # mask
    load_mask: bool = False
    mask_name: str = "gt_keep_masks/face/000000.png"
    mask_type: str = "random"  # "box" | "random" | "both" | "extreme"
    mask_len_range: List[int] = field(default_factory=lambda: [128, 129])
    mask_prob_range: List[float] = field(default_factory=lambda: [0.5, 0.5])

    # bookkeeping
    save_L: bool = True
    save_E: bool = True
    border: int = 0

    # LPIPS
    calc_LPIPS: bool = True

    sf: int = 1


@dataclass
class PnPInpaintHyperParams:
    """
    Hyper-parameters for a PnP inpainting baseline.

    Uses a generic denoiser prior (Gaussian smoothing or DRUNet) and a
    closed-form pixel-space data step enforcing consistency on known pixels.
    """

    num_iters: int = 50

    # denoiser
    denoiser: str = "gaussian"  # "gaussian", "drunet", or "diffbir"
    denoiser_sigma: Optional[float] = None  # if None -> use observation noise
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0

    # drunet
    drunet_weights_path: str = ""

    # diffbir — leave empty to download general_swinir_v1.ckpt from lxq007/DiffBIR automatically
    diffbir_weights_path: str = ""

    # output / metrics
    calc_LPIPS: bool = True
    border: int = 0
    save_L: bool = True
    save_E: bool = True
    clamp: bool = True


# ===========================================================================
# Hyper-parameter builders
# ===========================================================================


def _build_hparams_from_cfg(cfg: MethodConfig) -> DiffPIRInpaintHyperParams:
    """Create hyper-parameter object, allowing overrides through `cfg.extra`."""

    extra: Dict[str, Any] = cfg.extra or {}
    hp = DiffPIRInpaintHyperParams()
    hp.zeta_default = float(cfg.zeta)

    for key in [
        "noise_level_img",
        "noise_level_model",
        "model_name",
        "iter_num",
        "iter_num_U",
        "skip_type",
        "eta",
        "guidance_scale",
        "sub_1_analytic",
        "load_mask",
        "mask_name",
        "mask_type",
        "mask_len_range",
        "mask_prob_range",
        "calc_LPIPS",
    ]:
        if key in extra:
            setattr(hp, key, extra[key])

    if hp.noise_level_model is None:
        hp.noise_level_model = hp.noise_level_img

    return hp


def _build_pnp_hparams_from_cfg(cfg: MethodConfig) -> PnPInpaintHyperParams:
    """Create PnP hyper-parameter object, allowing overrides through `cfg.extra`."""
    extra: Dict[str, Any] = cfg.extra or {}
    hp = PnPInpaintHyperParams()
    for key in [
        "num_iters",
        "denoiser",
        "denoiser_sigma",
        "gaussian_kernel_size",
        "gaussian_sigma",
        "drunet_weights_path",
        "diffbir_weights_path",
        "calc_LPIPS",
        "border",
        "save_L",
        "save_E",
        "clamp",
    ]:
        if key in extra:
            setattr(hp, key, extra[key])
    return hp


# ===========================================================================
# Private helpers
# ===========================================================================


def _make_logger(name: str) -> logging.Logger:
    """Return a stream-handler logger for *name*, adding a handler only once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _build_mask(
    hp: DiffPIRInpaintHyperParams,
    img_H: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Return a float32 numpy mask of shape (H, W, C) with values in {0, 1}.

    When `hp.load_mask` is True the mask is loaded from `hp.mask_name`.
    Otherwise a random/box mask is generated with a fixed seed for reproducibility.
    """
    if hp.load_mask:
        n_channels = img_H.shape[2]
        mask = util.imread_uint(hp.mask_name, n_channels=n_channels).astype(bool)
        logger.info("Loaded mask from %s", hp.mask_name)
    else:
        gen = mask_generator(
            mask_type=hp.mask_type,
            mask_len_range=hp.mask_len_range,
            mask_prob_range=hp.mask_prob_range,
        )
        np.random.seed(seed=0)
        mask = gen(util.uint2tensor4(img_H)).numpy()
        mask = np.squeeze(mask)
        mask = np.transpose(mask, (1, 2, 0))
        logger.info(
            "Generated %s mask | len_range=%s | prob_range=%s",
            hp.mask_type,
            hp.mask_len_range,
            hp.mask_prob_range,
        )
    return mask.astype(np.float32)


def _make_inpaint_observation(
    img_H: np.ndarray,
    mask_np: np.ndarray,
    noise_level: float,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Apply the inpainting mask and AWGN to *img_H*.

    Returns:
        img_L      : float32 numpy array in [0, 1] (masked, for saving)
        y          : (1, C, H, W) torch tensor in [-1, 1] (for the diffusion loop)
        mask_tensor: (1, C, H, W) float32 torch tensor in {0, 1}
    """
    logger.info("Applying inpainting mask and AWGN (σ=%.4f)", noise_level)

    img_L = img_H * mask_np / 255.0  # [0, 1], known pixels only

    np.random.seed(seed=0)
    img_L = img_L * 2 - 1
    img_L += np.random.normal(0, noise_level * 2, img_L.shape)
    img_L = img_L / 2 + 0.5
    img_L = img_L * mask_np  # zero-out unknown pixels again

    y = util.single2tensor4(img_L).to(device)
    y = y * 2 - 1  # [-1, 1] for the diffusion loop

    mask_tensor = util.single2tensor4(mask_np).to(device)
    return img_L, y, mask_tensor


def _compute_lpips(
    loss_fn_vgg: Any,
    x_est: torch.Tensor,
    img_H: np.ndarray,
    device: torch.device,
) -> Optional[float]:
    """
    Compute LPIPS between *x_est* ∈ [0, 1] and uint8 ground-truth *img_H*.
    Returns None when *loss_fn_vgg* is None.
    """
    if loss_fn_vgg is None:
        return None
    img_H_t = np.transpose(img_H, (2, 0, 1))
    img_H_t = torch.from_numpy(img_H_t)[None, :, :, :].to(device)
    img_H_t = img_H_t / 255.0 * 2 - 1
    score = loss_fn_vgg(x_est.detach() * 2 - 1, img_H_t)
    return float(score.cpu().detach().numpy()[0][0][0][0])


def _save_outputs(
    img_E: np.ndarray,
    img_L: np.ndarray,
    method_out: str,
    img_name: str,
    ext: str,
    suffix: str,
    save_E: bool,
    save_L: bool,
    logger: logging.Logger,
) -> None:
    """Save the restored and/or masked images under *method_out*."""
    util.mkdir(method_out)
    if save_E:
        out_est = os.path.join(method_out, f"{img_name}_{suffix}{ext}")
        util.imsave(img_E, out_est)
        logger.info("Saved restored image to %s", out_est)
    if save_L:
        out_lr = os.path.join(method_out, f"{img_name}_masked{ext}")
        util.imsave(util.single2uint(img_L), out_lr)
        logger.info("Saved masked image to %s", out_lr)


# ===========================================================================
# Public method runners
# ===========================================================================


def run_diffpir_inpaint(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR inpainting runner.

    Mirrors `main_ddpir_inpainting.py` as a clean, single-image function:
    - generates (or loads) a binary mask
    - builds the masked/noisy observation *y*
    - runs the DiffPIR reverse-diffusion loop with the pixel-space
      closed-form data-consistency update
    - recovers known pixels from *y* at the end
    - writes the restored image into an `outputs` sub-folder
    - returns PSNR and LPIPS wrapped in `ImageResult`
    """

    assert (
        cfg.task == "inpaint"
    ), f"DiffPIR inpaint expects task='inpaint', got {cfg.task!r}"
    assert cfg.generate_mode == "DiffPIR", "run_diffpir_inpaint is specific to DiffPIR."

    hp = _build_hparams_from_cfg(cfg)
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"diffpir_inpaint.{img_name}")

    logger.info(
        "Starting DiffPIR inpainting | img=%s | lambda=%.3f | zeta=%.3f | noise=%.4f | model=%s",
        img_name,
        cfg.lambda_,
        cfg.zeta,
        hp.noise_level_img,
        hp.model_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Noise schedule
    beta_start = 0.1 / 1000
    beta_end = 20 / 1000
    betas = np.linspace(beta_start, beta_end, hp.num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    sigma = max(0.001, hp.noise_level_img)
    t_start = hp.num_train_timesteps - 1

    # Model loading
    model_zoo = os.path.join("", "model_zoo")
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
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    loss_fn_vgg = None
    if hp.calc_LPIPS:
        import lpips

        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    # Image, mask and degraded observation
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)

    mask_np = _build_mask(hp, img_H, logger)
    img_L, y, mask = _make_inpaint_observation(
        img_H, mask_np, hp.noise_level_img, device, logger
    )

    # Initialise x at t_start conditioned on y ([-1, 1])
    t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * hp.noise_level_img)
    sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * y + torch.sqrt(
        sqrt_1m_alphas_cumprod[t_start] ** 2
        - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y] ** 2
    ) * torch.randn_like(y)

    # Pre-compute sigmas / rhos
    sigmas, sigma_ks, rhos = [], [], []
    for i in range(hp.num_train_timesteps):
        sigmas.append(reduced_alpha_cumprod[hp.num_train_timesteps - 1 - i])
        sk = sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]
        sigma_ks.append(sk)
        rhos.append(cfg.lambda_ * (sigma**2) / (sk**2))
    rhos = torch.tensor(rhos).to(device)
    sigmas = torch.tensor(sigmas).to(device)

    # Time-step sequence
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

    # Reverse diffusion loop
    logger.info(
        "Starting reverse diffusion (%d steps, skip_type=%s)", len(seq), hp.skip_type
    )
    x_0 = None
    for i in tqdm(range(len(seq)), desc="DiffPIR Inpainting"):
        curr_sigma = sigmas[seq[i]].cpu().numpy()
        t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
        if t_i > t_start:
            continue

        for u in range(hp.iter_num_U):
            x0 = utils_model.model_fn(
                x,
                noise_level=curr_sigma * 255,
                model_out_type="pred_xstart",
                model_diffusion=model,
                diffusion=diffusion,
                ddim_sample=False,
                alphas_cumprod=alphas_cumprod,
            )

            # Pixel-space closed-form data-consistency step
            if seq[i] != seq[-1]:
                x0_p = (mask * y + rhos[t_i].float() * x0).div(mask + rhos[t_i])
                x0 = x0 + hp.guidance_scale * (x0_p - x0)

            # Step to next timestep
            if not (seq[i] == seq[-1] and u == hp.iter_num_U - 1):
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
                sqrt_ae = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                x = sqrt_ae * x + torch.sqrt(
                    sqrt_1m_alphas_cumprod[t_i] ** 2
                    - sqrt_ae**2 * sqrt_1m_alphas_cumprod[t_im1] ** 2
                ) * torch.randn_like(x)

        x_0 = x / 2 + 0.5

    assert x_0 is not None
    logger.info("Reverse diffusion finished for %s", img_name)

    # Recover known pixels from the observation
    x[mask.to(torch.bool)] = y[mask.to(torch.bool)]
    x_0 = x / 2 + 0.5

    # Metrics
    img_E = util.tensor2uint(x_0)
    psnr = util.calculate_psnr(img_E, img_H, border=hp.border)
    lpips_score = _compute_lpips(loss_fn_vgg, x_0, img_H, device)
    logger.info(
        "Results | img=%s | PSNR=%.3f dB | LPIPS=%s",
        img_name,
        psnr,
        f"{lpips_score:.4f}" if lpips_score is not None else "N/A",
    )

    # Save outputs
    extra = cfg.extra or {}
    method_out = os.path.join(extra.get("output_root", "outputs"), "diffpir_inpaint")
    _save_outputs(
        img_E, img_L, method_out, img_name, ext, "diffpir", hp.save_E, hp.save_L, logger
    )
    out_est = os.path.join(method_out, f"{img_name}_diffpir{ext}")

    return ImageResult(
        psnr=float(psnr), image_path=img_path, lpips=lpips_score, output_path=out_est
    )


def run_dps_inpaint(
    img_path: str, cfg: MethodConfig, mode: str = "DPS_y0"
) -> ImageResult:
    """
    Single-image DPS inpainting runner.

    `mode` should be either "DPS_y0" or "DPS_yt" and controls how the
    data-consistency term is applied in the loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hp = _build_hparams_from_cfg(cfg)
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"dps_inpaint.{img_name}")

    # 1. Noise schedule (Standard sequence)
    beta_start = 0.1 / 1000
    beta_end = 20 / 1000
    betas = np.linspace(beta_start, beta_end, hp.num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    betabar = 1.0 - alphas_cumprod

    # 2. Model loading (Safely FREEZING weights to prevent OOM)
    model_zoo = os.path.join("", "model_zoo")
    model_path = os.path.join(model_zoo, hp.model_name + ".pt")
    model_config = (
        dict(model_path=model_path, num_channels=128, num_res_blocks=1, attention_resolutions="16")
        if hp.model_name == "diffusion_ffhq_10m"
        else dict(model_path=model_path, num_channels=256, num_res_blocks=2, attention_resolutions="8,16,32")
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = True
    model = model.to(device)

    loss_fn_vgg = None
    if hp.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device) 
     
    # 3. Image + degradation loading
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)

    mask_np = _build_mask(hp, img_H, logger)
    # Note: _make_inpaint_observation returns 'y' already in the [-1, 1] domain!
    img_L, y, mask_tensor = _make_inpaint_observation(img_H, mask_np, hp.noise_level_img, device, logger)

    # 4. Build Forward Operator (A) for DPS Data Consistency
    def mask_operator(tensor):
        """Applies the binary mask to the tensor."""
        return tensor * mask_tensor

    # Initialize x at timestep T
    xt = torch.randn_like(y, device=device)

    logger.info("Starting DPS reverse diffusion (%d steps)", hp.num_train_timesteps)

    # 5. Reverse Diffusion Loop
    for i, t in tqdm(enumerate(reversed(range(hp.num_train_timesteps))), desc=f"DPS Inpaint {mode}"):
        
        # CRITICAL: Detach and require grad at the start of EVERY step
        xt = xt.detach().requires_grad_()
        t_tensor = torch.tensor([t], device=device)

        # 5a. Unet Prediction
        model_out = model(xt, t_tensor)
        eps = model_out[:, :3, :, :]

        # 5b. Data consistency (L2 Loss)
        if mode == "DPS_y0":
            # Calculate predicted clean image (xhat_0)
            xhat = (1.0 / sqrt_alphas_cumprod[t]) * xt - (sqrt_1m_alphas_cumprod[t] / sqrt_alphas_cumprod[t]) * eps
            l2 = torch.sum((mask_operator(xhat) - y) ** 2)
            
        elif mode == "DPS_yt":
            noise_y = mask_operator(eps.detach())
            y_t = (sqrt_alphas_cumprod[t] * y) + (sqrt_1m_alphas_cumprod[t] * noise_y)
            l2 = torch.sum((mask_operator(xt) - y_t) ** 2)

        grad_l2 = torch.autograd.grad(outputs=l2, inputs=xt)[0]

        # 5cStandard DDPM backward step (mu)
        mu = (1.0 / torch.sqrt(alphas[t])) * (xt - (betas[t] / torch.sqrt(betabar[t])) * eps)

        # DPS Gradient Correction
        zetat = 0.1 * torch.pow(l2, -0.5)

        # Final backward update combining DDPM, additive noise, and DPS guidance
        xt = (mu + torch.sqrt(betas[t]) * torch.randn_like(xt) - zetat * grad_l2).detach()

    # 6. Post-processing and Metrics
    # Standard DPS doesn't forcibly inject the exact known pixels back in, but for best metric performance in inpainting, it is standard practice.
    x_final = xt.detach() / 2.0 + 0.5
    y_0_1 = y / 2.0 + 0.5
    x_final = (x_final * (1 - mask_tensor)) + (y_0_1 * mask_tensor)
    x_0 = x_final.clamp(0.0, 1.0)

    logger.info("Reverse diffusion finished for %s", img_name)

    img_E = util.tensor2uint(x_0)
    psnr = util.calculate_psnr(img_E, img_H, border=hp.border)
    lpips_score = _compute_lpips(loss_fn_vgg, x_0, img_H, device)

    logger.info("Results | img=%s | PSNR=%.3f dB | LPIPS=%s", img_name, psnr, f"{lpips_score:.4f}" if lpips_score is not None else "N/A")

    extra = cfg.extra or {}
    method_out = os.path.join(extra.get("output_root", "outputs"), f"dps_{mode}_inpaint")
    _save_outputs(img_E, img_L, method_out, img_name, ext, f"dps_{mode}", hp.save_E, hp.save_L, logger)
    out_est = os.path.join(method_out, f"{img_name}_dps_{mode}{ext}")

    return ImageResult(psnr=float(psnr), image_path=img_path, lpips=lpips_score, output_path=out_est)


def run_pnp_inpaint(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play inpainting runner using DRUNet or another `Denoiser` prior.

    Uses a DPIR-style HQS iteration with:
      - Prior step: denoiser (DRUNet or Gaussian)
      - Data step: closed-form proximal on known pixels
        x = (mask * y + rho * x_denoised) / (mask + rho)
    """
    assert (
        cfg.task == "inpaint"
    ), f"PnP inpaint expects task='inpaint', got {cfg.task!r}"

    deg = _build_hparams_from_cfg(cfg)
    hp = _build_pnp_hparams_from_cfg(cfg)

    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"pnp_inpaint.{img_name}")

    logger.info(
        "Starting PnP inpainting | img=%s | denoiser=%s | iters=%d | mask=%s | noise=%.4f",
        img_name,
        hp.denoiser,
        hp.num_iters,
        deg.mask_type,
        deg.noise_level_img,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image, mask and degraded observation
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)

    mask_np = _build_mask(deg, img_H, logger)
    img_L, y_diffusion, mask_tensor = _make_inpaint_observation(
        img_H, mask_np, deg.noise_level_img, device, logger
    )

    # Work in [0, 1] space: convert y from [-1, 1] back to [0, 1]
    y = y_diffusion / 2 + 0.5

    # Denoiser selection
    denoiser: Denoiser
    if hp.denoiser.lower() == "drunet":
        denoiser = DRUNetDenoiser(weights_path=str(hp.drunet_weights_path))
        logger.info("Using DRUNet denoiser from %s", hp.drunet_weights_path)
    elif hp.denoiser.lower() == "diffbir":
        denoiser = DiffBIRDenoiser(weights_path=str(getattr(hp, "diffbir_weights_path", "")))
        logger.info("Using DiffBIR Stage-1 SwinIR denoiser (blind)")
    else:
        denoiser = GaussianDenoiser(
            kernel_size=int(hp.gaussian_kernel_size), sigma=float(hp.gaussian_sigma)
        )
        logger.info(
            "Using Gaussian denoiser (kernel=%d, sigma=%.2f)",
            hp.gaussian_kernel_size,
            hp.gaussian_sigma,
        )

    # DPIR-style rho/sigma schedule (logspace from modelSigma1 to modelSigma2)
    sigma_obs = max(0.255 / 255.0, float(deg.noise_level_img))
    model_sigma_2 = (
        float(hp.denoiser_sigma)
        if hp.denoiser_sigma is not None
        else float(deg.noise_level_img)
    )
    model_sigma_1 = 49.0
    sigma_schedule = (
        np.logspace(
            np.log10(model_sigma_1),
            np.log10(max(model_sigma_2 * 255.0, 0.01)),
            hp.num_iters,
        )
        / 255.0
    )
    rhos = [(sigma_obs**2) / (s**2) for s in sigma_schedule]
    rhos = torch.tensor(rhos, device=device, dtype=torch.float32)

    # Fill unknown pixels with the mean of known pixels so DRUNet gets
    # a natural-looking image from the very first iteration rather than an
    # image with hard zeros in 50 % of pixels.
    x = y.clone()
    unknown_mask = mask_tensor < 0.5
    if unknown_mask.any() and (~unknown_mask).any():
        known_mean = float(y[~unknown_mask].mean())
        x = torch.where(unknown_mask, torch.full_like(x, known_mean), x)

    # For zero (or near-zero) observation noise use a hard constraint:
    # after each denoiser step we force known pixels exactly back to y.
    # This avoids rho ≈ 0 letting denoiser artifacts bleed into known pixels.
    use_hard_constraint = float(deg.noise_level_img) < 1e-6

    for it in tqdm(range(hp.num_iters), desc="PnP Inpainting"):
        # Step 1: Prior (denoiser) step
        sigma_it = float(sigma_schedule[it])
        x = denoiser(x, sigma=sigma_it)

        # Step 2: Data step
        if use_hard_constraint:
            # Exact replacement of known pixels — equivalent to rho → ∞
            x = x * (1.0 - mask_tensor) + y * mask_tensor
        else:
            # Soft proximal: x = (mask * y + rho * x) / (mask + rho)
            rho = rhos[it].float()
            x = (mask_tensor * y + rho * x) / (mask_tensor + rho)

        if hp.clamp:
            x = x.clamp(0.0, 1.0)

    # Metrics
    loss_fn_vgg = None
    if hp.calc_LPIPS:
        import lpips

        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    img_E = util.tensor2uint(x)
    psnr = util.calculate_psnr(img_E, img_H, border=int(hp.border))
    lpips_score = _compute_lpips(loss_fn_vgg, x, img_H, device)

    logger.info(
        "Results | img=%s | PSNR=%.3f dB | LPIPS=%s",
        img_name,
        psnr,
        f"{lpips_score:.4f}" if lpips_score is not None else "N/A",
    )

    # Save outputs
    extra = cfg.extra or {}
    method_out = os.path.join(extra.get("output_root", "outputs"), "pnp_inpaint")
    suffix = f"pnp_{hp.denoiser}"
    _save_outputs(
        img_E,
        img_L,
        method_out,
        img_name,
        ext,
        suffix,
        hp.save_E,
        hp.save_L,
        logger,
    )
    out_est = os.path.join(method_out, f"{img_name}_{suffix}{ext}")

    return ImageResult(
        psnr=float(psnr), image_path=img_path, lpips=lpips_score, output_path=out_est
    )
