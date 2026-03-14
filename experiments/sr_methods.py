"""
Super-resolution method helpers for experiments.

These functions mirror the structure of `deblur_methods.py` but for the
super-resolution task, adapting the logic from `main_ddpir_sisr.py`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import cv2
import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils import utils_image as util
from utils import utils_model
from utils import utils_sisr as sr
from utils.utils_resizer import Resizer

from .common import ImageResult, MethodConfig
from .pnp_priors import GaussianDenoiser, DRUNetDenoiser, Denoiser


# ===========================================================================
# Hyper-parameter dataclasses
# ===========================================================================


@dataclass
class DiffPIRSRHyperParams:
    """
    Container for all DiffPIR super-resolution hyper-parameters.

    Defaults mirror `main_ddpir_sisr.py` so that experiments are
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

    # super-resolution
    sr_mode: str = "blur"  # "blur" or "cubic"
    classical_degradation: bool = False
    inIter: int = 1  # IBP inner iterations (cubic mode)
    gamma: float = 0.01  # IBP step size (cubic mode)

    # bookkeeping
    save_L: bool = True
    save_E: bool = True
    border: int = 0  # set dynamically to sf

    # LPIPS
    calc_LPIPS: bool = True


@dataclass
class PnPSRHyperParams:
    """
    Hyper-parameters for a PnP super-resolution baseline.

    Uses a denoiser prior (DRUNet or Gaussian) with a DPIR-style HQS iteration.
    The data step supports both the FFT Wiener-filter (blur mode) and
    iterative back-projection (cubic mode), matching the DiffPIR SR forward model.
    """

    # iterations
    num_iters: int = 50

    # denoiser
    denoiser: str = "gaussian"  # "gaussian" or "drunet"
    denoiser_sigma: Optional[float] = None  # terminal sigma; if None -> noise_level_img
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    drunet_weights_path: str = ""

    # SR degradation model (kept consistent with DiffPIR)
    sr_mode: str = "blur"  # "blur" or "cubic"
    classical_degradation: bool = False
    inIter: int = 1  # IBP inner iterations (cubic mode)
    gamma: float = 0.01  # IBP step size (cubic mode)

    # output / metrics
    calc_LPIPS: bool = True
    border: int = 0  # set dynamically to sf
    save_L: bool = True
    save_E: bool = True
    clamp: bool = True


# ===========================================================================
# Hyper-parameter builders
# ===========================================================================


def _build_hparams_from_cfg(cfg: MethodConfig) -> DiffPIRSRHyperParams:
    """Create hyper-parameter object, allowing overrides through `cfg.extra`."""

    extra: Dict[str, Any] = cfg.extra or {}
    hp = DiffPIRSRHyperParams()
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
        "sr_mode",
        "classical_degradation",
        "inIter",
        "gamma",
        "calc_LPIPS",
    ]:
        if key in extra:
            setattr(hp, key, extra[key])

    if hp.noise_level_model is None:
        hp.noise_level_model = hp.noise_level_img

    return hp


def _build_pnp_hparams_from_cfg(cfg: MethodConfig) -> PnPSRHyperParams:
    """Create PnP SR hyper-parameter object, allowing overrides through `cfg.extra`."""
    extra: Dict[str, Any] = cfg.extra or {}
    hp = PnPSRHyperParams()
    for key in [
        "num_iters",
        "denoiser",
        "denoiser_sigma",
        "gaussian_kernel_size",
        "gaussian_sigma",
        "drunet_weights_path",
        "sr_mode",
        "classical_degradation",
        "inIter",
        "gamma",
        "calc_LPIPS",
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


def _load_sr_kernel(sf: int, classical_degradation: bool, cwd: str = "") -> np.ndarray:
    """
    Load the SR degradation kernel from the kernels directory.

    For non-classical (bicubic) degradation the bicubic kernels are used;
    for classical degradation the 12-kernel set is used and the first kernel
    is selected. Returns a float64 numpy 2-D kernel.
    """
    if classical_degradation:
        kernels = hdf5storage.loadmat(os.path.join(cwd, "kernels", "kernels_12.mat"))[
            "kernels"
        ]
        k = kernels[0, 0].astype(np.float64)
    else:
        kernels = hdf5storage.loadmat(
            os.path.join(cwd, "kernels", "kernels_bicubicx234.mat")
        )["kernels"]
        k_index = sf - 2 if sf < 5 else 2
        k = kernels[0, k_index].astype(np.float64)
    return k


def _make_sr_observation(
    img_H: np.ndarray,
    k: np.ndarray,
    sf: int,
    hp: DiffPIRSRHyperParams,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Any, Any]:
    """
    Downsample *img_H* to create the LR observation and set up SR operators.

    Returns:
        img_L      : float32 numpy array in [0, 1] (LR with noise, for saving)
        y          : (1, C, h, w) torch tensor in [0, 1] (LR for data consistency)
        x_init     : (1, C, H, W) torch tensor in [-1, 1] (noisy bicubic upsampled init)
        down_sample: callable for the cubic downsampler (None for blur mode)
        up_sample  : callable for the cubic upsampler  (None for blur mode)
    """
    down_sample = None
    up_sample = None

    if hp.sr_mode == "blur":
        if hp.classical_degradation:
            img_L = sr.classical_degradation(img_H, k, sf)
            img_L = util.uint2single(img_L)
        else:
            img_L = util.imresize_np(util.uint2single(img_H), 1 / sf)

        np.random.seed(seed=0)
        img_L = img_L * 2 - 1
        img_L += np.random.normal(0, hp.noise_level_img * 2, img_L.shape)
        img_L = img_L / 2 + 0.5

        y = util.single2tensor4(img_L).to(device)

        # Bicubic upsample for initialisation
        x_np = cv2.resize(
            img_L,
            (img_L.shape[1] * sf, img_L.shape[0] * sf),
            interpolation=cv2.INTER_CUBIC,
        )
        if np.ndim(x_np) == 2:
            x_np = x_np[..., None]
        if hp.classical_degradation:
            x_np = sr.shift_pixel(x_np, sf)

    elif hp.sr_mode == "cubic":
        img_H_tensor = np.transpose(img_H, (2, 0, 1))
        img_H_tensor = torch.from_numpy(img_H_tensor)[None, :, :, :].to(device) / 255.0

        up_sample = partial(F.interpolate, scale_factor=sf)
        down_sample = Resizer(img_H_tensor.shape, 1 / sf).to(device)

        img_L_t = down_sample(img_H_tensor)
        img_L = img_L_t.cpu().numpy()
        img_L = np.squeeze(img_L)
        if img_L.ndim == 3:
            img_L = np.transpose(img_L, (1, 2, 0))

        np.random.seed(seed=0)
        img_L_noisy = img_L * 2 - 1
        img_L_noisy += np.random.normal(0, hp.noise_level_img * 2, img_L_noisy.shape)
        img_L_noisy = img_L_noisy / 2 + 0.5
        img_L = img_L_noisy

        y = util.single2tensor4(img_L).to(device)

        x_np = cv2.resize(
            img_L,
            (img_L.shape[1] * sf, img_L.shape[0] * sf),
            interpolation=cv2.INTER_CUBIC,
        )
        if np.ndim(x_np) == 2:
            x_np = x_np[..., None]

    else:
        raise ValueError(f"Unknown sr_mode {hp.sr_mode!r}. Use 'blur' or 'cubic'.")

    logger.info(
        "SR observation: mode=%s | sf=%d | LR size=%s | noise=%.4f",
        hp.sr_mode,
        sf,
        img_L.shape[:2],
        hp.noise_level_img,
    )
    return img_L, y, x_np, down_sample, up_sample


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
    sf: int,
    suffix: str,
    save_E: bool,
    save_L: bool,
    logger: logging.Logger,
) -> None:
    """Save the restored and/or LR images under *method_out*."""
    util.mkdir(method_out)
    if save_E:
        out_est = os.path.join(method_out, f"{img_name}_x{sf}_{suffix}{ext}")
        util.imsave(img_E, out_est)
        logger.info("Saved SR image to %s", out_est)
    if save_L:
        out_lr = os.path.join(method_out, f"{img_name}_x{sf}_LR{ext}")
        util.imsave(util.single2uint(img_L.squeeze()), out_lr)
        logger.info("Saved LR image to %s", out_lr)


# ===========================================================================
# Public method runners
# ===========================================================================


def run_diffpir_sr(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR super-resolution runner.

    Mirrors `main_ddpir_sisr.py` as a clean, single-image function:
    - loads the HR ground-truth and builds the LR observation
    - initialises *x* as a noisy bicubic upsampled version of the LR image
    - runs the DiffPIR reverse-diffusion loop with an FFT Wiener-filter (blur
      mode) or iterative back-projection (cubic mode) data-consistency step
    - returns PSNR, PSNR-Y, and LPIPS wrapped in `ImageResult`
    """

    assert cfg.task == "sr", f"DiffPIR SR expects task='sr', got {cfg.task!r}"
    assert cfg.generate_mode == "DiffPIR", "run_diffpir_sr is specific to DiffPIR."

    sf = cfg.sf
    hp = _build_hparams_from_cfg(cfg)
    hp.border = sf  # standard SR border exclusion

    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"diffpir_sr.{img_name}")

    logger.info(
        "Starting DiffPIR SR | img=%s | sf=%d | lambda=%.3f | zeta=%.3f | noise=%.4f | model=%s | mode=%s",
        img_name,
        sf,
        cfg.lambda_,
        cfg.zeta,
        hp.noise_level_img,
        hp.model_name,
        hp.sr_mode,
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

    # Load kernel (used for FFT data-consistency in blur mode)
    k = _load_sr_kernel(sf, hp.classical_degradation)
    logger.info(
        "Loaded SR kernel | classical=%s | sf=%d | shape=%s",
        hp.classical_degradation,
        sf,
        k.shape,
    )

    # Image and degraded observation
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)
    img_H = util.modcrop(img_H, sf)

    img_L, y, x_np, down_sample, up_sample = _make_sr_observation(
        img_H, k, sf, hp, device, logger
    )

    # Initialise x at t_start from bicubic-upsampled LR
    x = util.single2tensor4(x_np).to(device)
    x = sqrt_alphas_cumprod[t_start] * (2 * x - 1) + sqrt_1m_alphas_cumprod[
        t_start
    ] * torch.randn_like(x)

    # FFT pre-computation (blur mode) — done once, not per iteration
    k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

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
    for i in tqdm(range(len(seq)), desc=f"DiffPIR SR x{sf}"):
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

            # Data-consistency step
            if seq[i] != seq[-1]:
                tau = rhos[t_i].float().repeat(1, 1, 1, 1)

                if hp.sr_mode == "blur":
                    # Analytic Wiener-filter solution in Fourier domain
                    x0_p = (
                        sr.data_solution(
                            (x0 / 2 + 0.5).float(), FB, FBC, F2B, FBFy, tau, sf
                        )
                        * 2
                        - 1
                    )
                    x0 = x0 + hp.guidance_scale * (x0_p - x0)

                elif hp.sr_mode == "cubic":
                    # Iterative back-projection (IBP)
                    for _ in range(hp.inIter):
                        x0 = x0 / 2 + 0.5
                        x0 = x0 + hp.gamma * up_sample((y - down_sample(x0))) / (
                            1 + rhos[t_i]
                        )
                        x0 = x0 * 2 - 1

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

    # Metrics
    img_E = util.tensor2uint(x_0)
    psnr = util.calculate_psnr(img_E, img_H, border=hp.border)

    img_E_y = util.rgb2ycbcr(img_E, only_y=True)
    img_H_y = util.rgb2ycbcr(img_H, only_y=True)
    psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=hp.border)

    lpips_score = _compute_lpips(loss_fn_vgg, x_0, img_H, device)
    logger.info(
        "Results | img=%s | PSNR=%.3f dB | PSNR-Y=%.3f dB | LPIPS=%s",
        img_name,
        psnr,
        psnr_y,
        f"{lpips_score:.4f}" if lpips_score is not None else "N/A",
    )

    # Save outputs
    extra = cfg.extra or {}
    method_out = os.path.join(extra.get("output_root", "outputs"), "diffpir_sr")
    _save_outputs(
        img_E,
        img_L,
        method_out,
        img_name,
        ext,
        sf,
        "diffpir",
        hp.save_E,
        hp.save_L,
        logger,
    )
    out_est = os.path.join(method_out, f"{img_name}_x{sf}_diffpir{ext}")

    return ImageResult(
        psnr=float(psnr),
        psnr_y=float(psnr_y),
        image_path=img_path,
        lpips=lpips_score,
        output_path=out_est,
    )


def run_dps_sr(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult:
    """
    Single-image DPS SR runner.

    `mode` should be either "DPS_y0" or "DPS_yt" and controls how the
    data-consistency term is applied in the loop.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sf = cfg.sf
    hp = _build_hparams_from_cfg(cfg)
    hp.border = sf 
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"dps_sr.{img_name}")

    # 1. Noise schedule
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

    if hp.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
    else:
        loss_fn_vgg = None

    # 3. Image + degradation loading
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)
    img_H = util.modcrop(img_H, sf)

    k = _load_sr_kernel(sf, hp.classical_degradation)
    img_L, y, x_np, down_sample, up_sample = _make_sr_observation(img_H, k, sf, hp, device, logger)
    
    # Scale LR observation y to [-1, 1]
    y_scaled = y * 2.0 - 1.0

    # 4. Build Forward Operator (A)
    if hp.sr_mode == "blur":
        # Pure spatial differentiable blur + strided downsample
        k_tensor = torch.tensor(k, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        pad_sz = k.shape[0] // 2
        def forward_operator(tensor):
            x_pad = torch.nn.functional.pad(tensor, (pad_sz, pad_sz, pad_sz, pad_sz), mode="circular")
            # Convolution with stride effectively combines blur and downsampling
            return torch.nn.functional.conv2d(x_pad, k_tensor, stride=sf, groups=3)
    elif hp.sr_mode == "cubic":
        def forward_operator(tensor):
            return down_sample(tensor)

    # Initialize xt at HR resolution
    B, C, h, w = y.shape
    xt = torch.randn((B, C, h * sf, w * sf), device=device)

    logger.info("Starting DPS reverse diffusion (%d steps)", hp.num_train_timesteps)

    # 5. Reverse Diffusion Loop
    for i, t in tqdm(enumerate(reversed(range(hp.num_train_timesteps))), desc=f"DPS SR x{sf} {mode}"):
        
        # CRITICAL: Detach and require grad at the start of EVERY step
        xt = xt.detach().requires_grad_()
        t_tensor = torch.tensor([t], device=device)

        # 5a. Unet Prediction
        model_out = model(xt, t_tensor)
        eps = model_out[:, :3, :, :]

        # Calculate predicted clean image (xhat_0)
        xhat = (1.0 / sqrt_alphas_cumprod[t]) * xt - (sqrt_1m_alphas_cumprod[t] / sqrt_alphas_cumprod[t]) * eps

        # 5b. Data consistency (L2 Loss)
        if mode == "DPS_y0":
            loss = torch.sum((forward_operator(xhat) - y_scaled) ** 2)
            
        elif mode == "DPS_yt":
            noise_y = forward_operator(eps.detach())
            y_t = (sqrt_alphas_cumprod[t] * y_scaled) + (sqrt_1m_alphas_cumprod[t] * noise_y)
            loss = torch.sum((forward_operator(xt) - y_t) ** 2)

        grad_l2 = torch.autograd.grad(outputs=loss, inputs=xt)[0]

        # 5c. Standard DDPM backward step (mu)
        mu = (1.0 / torch.sqrt(alphas[t])) * (xt - (betas[t] / torch.sqrt(betabar[t])) * eps)

        # 5d. DPS Gradient Correction
        loss_val = loss.detach().item()
        zetat = cfg.lambda_ * (loss_val ** -0.5) if loss_val > 0 else 0.0
        noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

        xt = mu + torch.sqrt(betas[t]) * noise - zetat * grad_l2

    # 6. Post-processing and Metrics
    x_0 = xt.detach() / 2.0 + 0.5
    x_0 = x_0.clamp(0.0, 1.0)

    logger.info("Reverse diffusion finished for %s", img_name)

    img_E = util.tensor2uint(x_0)
    psnr = util.calculate_psnr(img_E, img_H, border=hp.border)
    img_E_y = util.rgb2ycbcr(img_E, only_y=True)
    img_H_y = util.rgb2ycbcr(img_H, only_y=True)
    psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=hp.border)
    lpips_score = _compute_lpips(loss_fn_vgg, x_0, img_H, device)

    logger.info("Results | img=%s | PSNR=%.3f dB | PSNR-Y=%.3f dB | LPIPS=%s", img_name, psnr, psnr_y, f"{lpips_score:.4f}" if lpips_score is not None else "N/A")

    extra = cfg.extra or {}
    method_out = os.path.join(extra.get("output_root", "outputs"), f"dps_{mode}_sr")
    _save_outputs(img_E, img_L, method_out, img_name, ext, sf, f"dps_{mode}", hp.save_E, hp.save_L, logger)
    out_est = os.path.join(method_out, f"{img_name}_x{sf}_dps_{mode}{ext}")

    return ImageResult(psnr=float(psnr), psnr_y=float(psnr_y), image_path=img_path, lpips=lpips_score, output_path=out_est)


def run_pnp_sr(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play super-resolution runner using DRUNet or another `Denoiser` prior.

    Uses a DPIR-style HQS iteration with:
      - Prior step: DRUNet (or Gaussian) denoiser on the HR estimate
      - Data step (blur mode): analytic FFT Wiener-filter solution via
        `sr.pre_calculate` / `sr.data_solution` — identical to DiffPIR SR
      - Data step (cubic mode): iterative back-projection (IBP)
    """
    assert cfg.task == "sr", f"PnP SR expects task='sr', got {cfg.task!r}"

    sf = cfg.sf
    deg = _build_hparams_from_cfg(cfg)
    hp = _build_pnp_hparams_from_cfg(cfg)
    hp.border = sf

    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"pnp_sr.{img_name}")

    logger.info(
        "Starting PnP SR | img=%s | sf=%d | denoiser=%s | iters=%d | mode=%s | noise=%.4f",
        img_name,
        sf,
        hp.denoiser,
        hp.num_iters,
        hp.sr_mode,
        deg.noise_level_img,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image and degraded observation (same pipeline as DiffPIR for fair comparison)
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)
    img_H = util.modcrop(img_H, sf)

    k = _load_sr_kernel(sf, hp.classical_degradation)
    logger.info(
        "Loaded SR kernel | classical=%s | sf=%d | shape=%s",
        hp.classical_degradation,
        sf,
        k.shape,
    )

    img_L, y, x_np, down_sample, up_sample = _make_sr_observation(
        img_H, k, sf, deg, device, logger
    )

    # Denoiser selection
    denoiser: Denoiser
    if hp.denoiser.lower() == "drunet":
        denoiser = DRUNetDenoiser(weights_path=str(hp.drunet_weights_path))
        logger.info("Using DRUNet denoiser from %s", hp.drunet_weights_path)
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
            np.log10(max(model_sigma_2 * 255.0, 0.255)),
            hp.num_iters,
        )
        / 255.0
    )
    rhos = [(sigma_obs**2) / (s**2) for s in sigma_schedule]
    rhos = torch.tensor(rhos, device=device, dtype=torch.float32)

    # Initialize x at HR size from bicubic-upsampled LR
    x = util.single2tensor4(x_np).to(device)

    # FFT pre-computation for blur-mode data step (done once outside the loop)
    FB, FBC, F2B, FBFy = None, None, None, None
    if hp.sr_mode == "blur":
        k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
        FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

    for it in tqdm(range(hp.num_iters), desc=f"PnP SR x{sf}"):
        # Step 1: Prior (denoiser) step
        sigma_it = float(sigma_schedule[it])
        x = denoiser(x, sigma=sigma_it)

        # Step 2: Data step
        rho = rhos[it].float()

        if hp.sr_mode == "blur":
            tau = rho.repeat(1, 1, 1, 1)
            x = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf)
        elif hp.sr_mode == "cubic":
            for _ in range(hp.inIter):
                x = x + hp.gamma * up_sample(y - down_sample(x)) / (1.0 + rho)
        else:
            raise ValueError(f"Unknown sr_mode {hp.sr_mode!r}")

        if hp.clamp:
            x = x.clamp(0.0, 1.0)

    # Metrics
    loss_fn_vgg = None
    if hp.calc_LPIPS:
        import lpips

        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    img_E = util.tensor2uint(x)
    psnr = util.calculate_psnr(img_E, img_H, border=int(hp.border))

    img_E_y = util.rgb2ycbcr(img_E, only_y=True)
    img_H_y = util.rgb2ycbcr(img_H, only_y=True)
    psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=int(hp.border))

    lpips_score = _compute_lpips(loss_fn_vgg, x, img_H, device)

    logger.info(
        "Results | img=%s | PSNR=%.3f dB | PSNR-Y=%.3f dB | LPIPS=%s",
        img_name,
        psnr,
        psnr_y,
        f"{lpips_score:.4f}" if lpips_score is not None else "N/A",
    )

    # Save outputs
    extra = cfg.extra or {}
    method_out = os.path.join(extra.get("output_root", "outputs"), "pnp_sr")
    suffix = f"pnp_{hp.denoiser}"
    _save_outputs(
        img_E,
        img_L,
        method_out,
        img_name,
        ext,
        sf,
        suffix,
        hp.save_E,
        hp.save_L,
        logger,
    )
    out_est = os.path.join(method_out, f"{img_name}_x{sf}_{suffix}{ext}")

    return ImageResult(
        psnr=float(psnr),
        psnr_y=float(psnr_y),
        image_path=img_path,
        lpips=lpips_score,
        output_path=out_est,
    )
