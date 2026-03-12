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
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
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
from .pnp_priors import GaussianDenoiser, DRUNetDenoiser, Denoiser


# ===========================================================================
# Hyper-parameter dataclasses
# ===========================================================================


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


@dataclass
class PnPDeblurHyperParams:
    """
    Hyper-parameters for a simple PnP deblurring baseline.

    This uses a generic denoiser prior (Gaussian smoothing by default) and a
    gradient descent data step with the circular-convolution blur operator so
    it matches the `mode="wrap"` degradation used in DiffPIR.
    """

    # iterations
    num_iters: int = 50
    step_size: float = 1.0
    data_weight: float = 1.0

    # denoiser
    denoiser: str = "gaussian"  # "gaussian" or "drunet"
    denoiser_sigma: Optional[float] = None  # if None -> use observation noise
    sigma_schedule: str = "constant"  # "constant" or "linear"

    # gaussian denoiser params (fallback)
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0

    # drunet params (only used if implemented in this repo)
    drunet_weights_path: str = ""

    # output / metrics
    calc_LPIPS: bool = True
    border: int = 0
    save_L: bool = True
    save_E: bool = True
    clamp: bool = True


# ===========================================================================
# Hyper-parameter builders
# ===========================================================================


def _build_hparams_from_cfg(cfg: MethodConfig) -> DiffPIRDeblurHyperParams:
    """Create hyper-parameter object, allowing overrides through `cfg.extra`."""

    extra: Dict[str, Any] = cfg.extra or {}
    hp = DiffPIRDeblurHyperParams()
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
        "use_DIY_kernel",
        "blur_mode",
        "kernel_size",
        "kernel_std",
        "calc_LPIPS",
    ]:
        if key in extra:
            setattr(hp, key, extra[key])

    if hp.noise_level_model is None:
        hp.noise_level_model = hp.noise_level_img

    return hp


def _build_pnp_hparams_from_cfg(cfg: MethodConfig) -> PnPDeblurHyperParams:
    extra: Dict[str, Any] = cfg.extra or {}
    hp = PnPDeblurHyperParams()
    for key in [
        "num_iters",
        "step_size",
        "data_weight",
        "denoiser",
        "denoiser_sigma",
        "sigma_schedule",
        "gaussian_kernel_size",
        "gaussian_sigma",
        "drunet_weights_path",
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
# Private helpers (shared by both methods)
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


def _build_blur_kernel(
    hp: DiffPIRDeblurHyperParams,
    device: torch.device,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Build and return the 2-D blur kernel *k* (numpy, float32).

    Mirrors the DIY-kernel branch of `main_ddpir_deblur.py` exactly,
    including the fixed random seed so all methods see the same blur.
    """
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
        k = np.squeeze(
            kernel.get_kernel().to(device, dtype=torch.float).detach().cpu().numpy()
        )
    else:
        import hdf5storage

        kernels = hdf5storage.loadmat(os.path.join("", "kernels", "Levin09.mat"))[
            "kernels"
        ]
        k = kernels[0, 0].astype(np.float32)

    logger.info(
        "Blur kernel: mode=%s | kernel_size=%d | kernel_std≈%.3f",
        hp.blur_mode,
        hp.kernel_size,
        hp.kernel_std,
    )
    return k


def _make_noisy_observation(
    img_H: np.ndarray,
    k: np.ndarray,
    noise_level: float,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Apply circular-convolution blur + AWGN to *img_H*.

    Returns:
        img_L : float32 numpy array in [0, 1]  (for saving)
        y     : (1, C, H, W) torch tensor on *device* in [0, 1]
    """
    logger.info("Applying blur and AWGN (σ≈%.4f)", noise_level)
    img_L = ndimage.convolve(img_H, np.expand_dims(k, axis=2), mode="wrap")
    img_L = util.uint2single(img_L)
    np.random.seed(seed=0)
    img_L = img_L * 2 - 1
    img_L += np.random.normal(0, noise_level * 2, img_L.shape)
    img_L = img_L / 2 + 0.5

    y = util.single2tensor4(img_L).to(device)
    return img_L, y


def _compute_lpips(
    loss_fn_vgg: Any,
    x_est: torch.Tensor,
    img_H: np.ndarray,
    device: torch.device,
) -> Optional[float]:
    """
    Compute LPIPS between the estimated image *x_est* ∈ [0,1] and uint8
    ground-truth *img_H*.  Returns None when *loss_fn_vgg* is None.
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
    """Save the restored and/or LR images under *method_out*."""
    util.mkdir(method_out)
    if save_E:
        out_est = os.path.join(method_out, f"{img_name}_{suffix}{ext}")
        util.imsave(img_E, out_est)
        logger.info("Saved restored image to %s", out_est)
    if save_L:
        out_lr = os.path.join(method_out, f"{img_name}_LR{ext}")
        util.imsave(util.single2uint(img_L), out_lr)
        logger.info("Saved LR image to %s", out_lr)


# ===========================================================================
# Public method runners
# ===========================================================================


def run_diffpir_deblur(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR deblurring runner.

    This is a direct, single-image analogue of `main_ddpir_deblur.py`:
    - builds the blur kernel and noisy observation from the sharp GT image
    - runs the DiffPIR sampling loop
    - writes the restored image into an `outputs` sub-folder
    - returns PSNR and LPIPS wrapped in `ImageResult`
    """

    assert (
        cfg.task == "deblur"
    ), f"DiffPIR deblur expects task='deblur', got {cfg.task!r}"
    assert cfg.generate_mode == "DiffPIR", "run_diffpir_deblur is specific to DiffPIR."

    hp = _build_hparams_from_cfg(cfg)
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"diffpir_deblur.{img_name}")

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

    # Noise schedule (same construction as `main_ddpir_deblur.py`)
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

    # Image + degradation
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)
    img_H = util.modcrop(img_H, 8)

    k = _build_blur_kernel(hp, device, logger)
    img_L, y = _make_noisy_observation(img_H, k, hp.noise_level_img, device, logger)

    # Initialise x at t_start conditioned on y
    t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * hp.noise_level_img)
    sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * (2 * y - 1) + torch.sqrt(
        sqrt_1m_alphas_cumprod[t_start] ** 2
        - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y] ** 2
    ) * torch.randn_like(y)

    # FFT pre-computation (sf=1 for deblurring: no upscaling)
    hp.sf = 1
    k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, hp.sf)

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
    for i in tqdm(range(len(seq)), desc="DiffPIR Deblurring"):
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

            # Data-consistency (analytic FFT step)
            if seq[i] != seq[-1]:
                tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                x0_p = (
                    sr.data_solution(
                        (x0 / 2 + 0.5).float(), FB, FBC, F2B, FBFy, tau, hp.sf
                    )
                    * 2
                    - 1
                )
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
    method_out = os.path.join(extra.get("output_root", "outputs"), "diffpir_deblur")
    _save_outputs(
        img_E, img_L, method_out, img_name, ext, "diffpir", hp.save_E, hp.save_L, logger
    )
    out_est = os.path.join(method_out, f"{img_name}_diffpir{ext}")

    return ImageResult(psnr=float(psnr), image_path=img_path, lpips=lpips_score, output_path=out_est)


def run_dps_deblur(
    img_path: str, cfg: MethodConfig, mode: str = "DPS_y0"
) -> ImageResult:
    """
    Single-image DPS deblurring runner (to be implemented).

    `mode` should be either "DPS_y0" or "DPS_yt" and controls how the
    data-consistency term is applied in the loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hp = _build_hparams_from_cfg(cfg)
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"diffpir_deblur.{img_name}")

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

    # 2. Model loading (Freezing parameters, but we will require grad on inputs)
    model_zoo = os.path.join("", "model_zoo")
    model_path = os.path.join(model_zoo, hp.model_name + ".pt")
    model_config = (
        dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
            use_checkpoint=False
        )
        if hp.model_name == "diffusion_ffhq_10m"
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
            use_checkpoint=False
        )
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
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
    img_H = util.modcrop(img_H, 8)

    k = _build_blur_kernel(hp, device, logger)
    img_L, y = _make_noisy_observation(img_H, k, hp.noise_level_img, device, logger)

    # 4. Build Forward Operator (A) for DPS Data Consistency
    # Adjust kernel shape for depthwise 2D convolution over 3 channels
    k_tensor = torch.tensor(k, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    k_tensor = k_tensor.repeat(3, 1, 1, 1)

    def linear_operator(x_hat):
        """Applies circular blur matching the degradation process."""
        # Map x_hat from [-1, 1] to [0, 1] to match the y tensor's domain
        x_0_1 = x_hat / 2.0 + 0.5
        pad_sz = k.shape[0] // 2
        # Use circular padding to mimic scipy's mode="wrap"
        x_pad = torch.nn.functional.pad(x_0_1, (pad_sz, pad_sz, pad_sz, pad_sz), mode='circular')
        return torch.nn.functional.conv2d(x_pad, k_tensor, groups=3)

    # Initialize x at timestep T
    xt = torch.randn_like(y, device=device)

    logger.info("Starting DPS reverse diffusion (%d steps)", hp.num_train_timesteps)

    # 5. Reverse Diffusion Loop (Inspired by tp9 notebook)
    for i, t_val in tqdm(enumerate(reversed(range(hp.num_train_timesteps))), desc="DPS Deblurring"):
        
        # We need gradients flowing back to xt to compute the data consistency step
        xt = xt.requires_grad_()
        t_tensor = torch.tensor([t_val], device=device)

        # 5a. Unet Prediction
        model_out = model(xt, t_tensor)
        eps = model_out[:, :3, :, :]
        
        # Calculate predicted clean image (xhat_0)
        xhat = (1.0 / sqrt_alphas_cumprod[t_val]) * xt - (sqrt_1m_alphas_cumprod[t_val] / sqrt_alphas_cumprod[t_val]) * eps

       # 5b. Data consistency (L2 Loss & Gradient)
        if mode == "DPS_y0":
            # Standard DPS: Compare predicted clean image (mapped to [0,1]) to clean measurement
            target_x = xhat
            loss = torch.sum((linear_operator(target_x) - y)**2)

        elif mode == "DPS_yt":
            # 1. Create a pure convolution operator (without the [-1,1] to [0,1] shift)
            # This allows us to accurately scale the noise in the raw diffusion space
            pad_sz = k.shape[0] // 2
            def pure_conv(tensor):
                x_pad = torch.nn.functional.pad(tensor, (pad_sz, pad_sz, pad_sz, pad_sz), mode='circular')
                return torch.nn.functional.conv2d(x_pad, k_tensor, groups=3)
            
            # 2. Scale the clean measurement 'y' to the [-1, 1] diffusion space
            y_scaled = y * 2.0 - 1.0 
            
            # 3. Construct the consistent noisy measurement y_t
            # THE FIX: Use the model's own predicted noise (eps) instead of random noise.
            # We detach it because y_t is our fixed target for this step.
            noise_y = pure_conv(eps.detach())
            y_t = (sqrt_alphas_cumprod[t_val] * y_scaled) + (sqrt_1m_alphas_cumprod[t_val] * noise_y)
            
            # 4. Compare the purely blurred noisy image to the aligned noisy measurement
            pred_yt = pure_conv(xt)
            loss = torch.sum((pred_yt - y_t)**2)

        grad_l2 = torch.autograd.grad(outputs=loss, inputs=xt)[0]

        # 5c. Standard DDPM backward step (mu)
        mu = (1.0 / torch.sqrt(alphas[t_val])) * (xt - (betas[t_val] / torch.sqrt(betabar[t_val])) * eps)

        # 5d. DPS Gradient Correction
        # The scaling factor zeta_t is adapted directly from the notebook
        zetat = 0.1 * torch.pow(loss.detach(), -0.5) if loss.item() > 0 else 0.0
        
        if t_val > 0:
            noise = torch.randn_like(xt) * torch.sqrt(betas[t_val])
        else:
            noise = 0.0

        # Final backward update combining DDPM, additive noise, and DPS guidance
        xt = (mu + noise - zetat * grad_l2).detach()

    # 6. Post-processing and Metrics
    x_0 = xt / 2.0 + 0.5
    x_0 = x_0.clamp(0.0, 1.0)
    
    logger.info("Reverse diffusion finished for %s", img_name)

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
    method_out = os.path.join(extra.get("output_root", "outputs"), f"dps_{mode}_deblur")
    _save_outputs(
        img_E, img_L, method_out, img_name, ext, f"dps_{mode}", hp.save_E, hp.save_L, logger
    )
    out_est = os.path.join(method_out, f"{img_name}_dps_{mode}{ext}")

    return ImageResult(psnr=float(psnr), image_path=img_path, lpips=lpips_score, output_path=out_est)


def run_pnp_drunet_deblur(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play deblurring runner using DRUNet or another `Denoiser` prior.

    Uses a gradient-descent data step with the same circular-convolution
    degradation as DiffPIR for fair comparison.
    """
    assert cfg.task == "deblur", f"PnP deblur expects task='deblur', got {cfg.task!r}"

    deg = _build_hparams_from_cfg(cfg)
    hp = _build_pnp_hparams_from_cfg(cfg)

    img_name, ext = os.path.splitext(os.path.basename(img_path))
    logger = _make_logger(f"pnp_deblur.{img_name}")

    logger.info(
        "Starting PnP deblurring | img=%s | denoiser=%s | iters=%d | step=%.3f | blur=%s | noise=%.4f",
        img_name,
        hp.denoiser,
        hp.num_iters,
        hp.step_size,
        deg.blur_mode,
        deg.noise_level_img,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image + degradation (same as DiffPIR for fair comparison)
    n_channels = 3
    logger.info("Loading ground-truth image from %s", img_path)
    img_H = util.imread_uint(img_path, n_channels=n_channels)
    img_H = util.modcrop(img_H, 8)

    k = _build_blur_kernel(deg, device, logger)
    img_L, y = _make_noisy_observation(img_H, k, deg.noise_level_img, device, logger)

    # Denoiser selection
    denoiser: Denoiser
    if hp.denoiser.lower() == "drunet":
        denoiser = DRUNetDenoiser(weights_path=str(hp.drunet_weights_path))
        logger.info("Using DRUNet denoiser from %s", hp.drunet_weights_path)
    else:
        denoiser = GaussianDenoiser(
            kernel_size=int(hp.gaussian_kernel_size), sigma=float(hp.gaussian_sigma)
        )

    # DPIR-style rho/sigma schedule (logspace from modelSigma1 to modelSigma2)
    sigma_obs = max(0.255 / 255.0, float(deg.noise_level_img))
    model_sigma_2 = (
        float(hp.denoiser_sigma)
        if hp.denoiser_sigma is not None
        else float(deg.noise_level_img)
    )
    model_sigma_1 = 49.0  # DPIR default
    sigma_schedule = (
        np.logspace(
            np.log10(model_sigma_1), np.log10(model_sigma_2 * 255.0), hp.num_iters
        )
        / 255.0
    )
    rhos = [(sigma_obs**2) / (s**2) / 3.0 for s in sigma_schedule]
    rhos = torch.tensor(rhos, device=device, dtype=torch.float32)

    # FFT pre-computation for analytic data step (sf=1 for deblur)
    deblur_sf = 1
    k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, deblur_sf)

    x = y.clone()

    for it in tqdm(range(hp.num_iters), desc="PnP Deblurring"):
        # Data step: analytic FFT solution (DPIR-style) instead of gradient descent
        tau = rhos[it].float().repeat(1, 1, 1, 1)
        x = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, deblur_sf)

        # Prior step: denoiser
        sigma_it = float(sigma_schedule[it])
        x = denoiser(x, sigma=sigma_it)

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
    method_out = os.path.join(extra.get("output_root", "outputs"), "pnp_deblur")
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

    return ImageResult(psnr=float(psnr), image_path=img_path, lpips=lpips_score, output_path=out_est)
