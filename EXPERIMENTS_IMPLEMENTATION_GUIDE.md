## Experiments implementation guide

This guide is a **concrete TODO list** for finishing the comparison framework.
It tells you **exactly what to implement**, **where**, and **how each function is called**.

---

### 0. Mental model

- We have **per-image methods** (DiffPIR, DPS, PnP+DRUNet) for each task:
  - `sr` (super-resolution)
  - `deblur`
  - `inpaint`
- Each per-image method has the signature:
  - `method(img_path: str, cfg: MethodConfig) -> ImageResult`
- `experiments.common.run_experiment(...)` calls these per-image methods over a list of image paths and aggregates metrics into a `RunResult`.
- The **notebook** and any future scripts will only need to:
  - Choose `task` and `methods` (e.g. `["diffpir", "dps_y0", "pnp_drunet"]`).
  - Call `compare_task(...)` from the notebook (already wired) or their own loop.

Everything below is about making those per-image methods and DRUNet work.

---

### 1. `experiments/common.py` – already done

You **do not need to change this file** for the core project, but you should know how to use it.

- **Key types**:
  - `MethodConfig`
  - `ImageResult`
  - `RunResult`
- **Key helpers**:
  - `load_image_paths(testset_root: str) -> List[str]`
  - `run_experiment(method_config: MethodConfig, image_paths: Sequence[str], method_fn, output_root: Optional[str]) -> RunResult`

**How methods will be called**:

```python
from experiments.common import MethodConfig, run_experiment, load_image_paths

def my_method(img_path: str, cfg: MethodConfig) -> ImageResult:
    ...

cfg = MethodConfig(
    name="diffpir",
    task="sr",
    generate_mode="DiffPIR",
    lambda_=1.0,
    zeta=0.25,
    sf=4,
)
paths = load_image_paths("testsets/demo_test")
result = run_experiment(method_config=cfg, image_paths=paths, method_fn=my_method)
```

---

### 2. Super‑resolution methods – `experiments/sr_methods.py`

**Goal**: Implement DiffPIR, DPS, and PnP+DRUNet **per-image SR methods**.

Current stubs:

- `run_diffpir_sr(img_path: str, cfg: MethodConfig) -> ImageResult`
- `run_dps_sr(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult`
- `run_pnp_drunet_sr(img_path: str, cfg: MethodConfig) -> ImageResult`

These are imported and used by:

- The notebook (`experiments/comparison_notebook.ipynb`), via the SR method registry.

#### 2.1 TODO – `run_diffpir_sr`

**What to do**:

- Mirror the **SR branch** of `main_ddpir.py` (or `main_ddpir_sisr.py`) but for a **single image**:
  1. Load HR image from `img_path` with `utils.utils_image.imread_uint`.
  2. Modcrop and generate LR observation (blur or bicubic + noise) like in the SR code.
  3. Build the noise schedule (`betas`, `alphas_cumprod`, etc.).
  4. Run the DiffPIR sampling loop with:
     - `cfg.generate_mode == "DiffPIR"`
     - scalar `cfg.lambda_`, `cfg.zeta`, `cfg.sf`
  5. Produce a reconstructed image tensor `x_0` in \([0,1]\).
  6. Convert to uint image and compute metrics with `utils.utils_image`:
     - `psnr = calculate_psnr(...)`
     - `psnr_y = calculate_psnr(...)` on Y channel
     - optional LPIPS if you want (`lpips` package already in requirements).
  7. Return:

     ```python
     return ImageResult(psnr=float(psnr), psnr_y=float(psnr_y), lpips=lpips_value_or_None)
     ```

**How it will be called** (from the notebook or other code):

```python
from experiments.sr_methods import run_diffpir_sr
from experiments.common import MethodConfig

cfg = MethodConfig(
    name="diffpir",
    task="sr",
    generate_mode="DiffPIR",
    lambda_=1.0,
    zeta=0.25,
    sf=4,
)
res = run_diffpir_sr("testsets/demo_test/69037.png", cfg)
print(res.psnr)
```

#### 2.2 TODO – `run_dps_sr`

**What to do**:

- Reuse almost all of the logic from `run_diffpir_sr`, but:
  - Set **DPS-style generate mode** by `mode` argument:
    - `"DPS_y0"` → match behavior of `generate_mode == "DPS_y0"` in `main_ddpir.py`.
    - `"DPS_yt"` → match behavior of `generate_mode == "DPS_yt"`.
  - In the data-consistency step, call `utils_model.grad_and_value` exactly like the DPS case in `main_ddpir.py`.

**How it will be called**:

```python
from experiments.sr_methods import run_dps_sr

cfg = MethodConfig(
    name="dps_y0",
    task="sr",
    generate_mode="DPS_y0",
    lambda_=1.0,
    zeta=0.25,
    sf=4,
)
res = run_dps_sr("testsets/demo_test/69037.png", cfg, mode="DPS_y0")
```

The notebook already wraps this in lambdas like:

```python
"dps_y0": lambda img, cfg: sr_methods.run_dps_sr(img, cfg, mode="DPS_y0")
```

#### 2.3 TODO – `run_pnp_drunet_sr`

**What to do**:

- Implement a **standard iterative plug‑and‑play SR solver**:
  1. Build the forward operator for SR (downsample by `cfg.sf`) using `utils.utils_sisr` and/or `Resizer`.
  2. Define a simple data-fidelity step (e.g. gradient descent on \\(\\|H(x) - y\\|^2\\)).
  3. For the prior step, call a denoiser:
     - Initially, you can use `GaussianDenoiser` from `experiments.pnp_priors`.
     - Later, switch to `DRUNetDenoiser` once implemented.
  4. Iterate data_step → prior_step several times.
  5. Compute metrics like in `run_diffpir_sr` and return an `ImageResult`.

**How it will be called**:

```python
from experiments.sr_methods import run_pnp_drunet_sr
cfg = MethodConfig(name="pnp_drunet", task="sr", generate_mode="pnp_drunet", lambda_=1.0, zeta=0.25, sf=4)
res = run_pnp_drunet_sr("testsets/demo_test/69037.png", cfg)
```

The notebook will call this when you include `"pnp_drunet"` in the SR method list.

---

### 3. Deblur methods – `experiments/deblur_methods.py`

Stubs:

- `run_diffpir_deblur(img_path: str, cfg: MethodConfig) -> ImageResult`
- `run_dps_deblur(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult`
- `run_pnp_drunet_deblur(img_path: str, cfg: MethodConfig) -> ImageResult`

These are wired into the notebook through `get_deblur_methods()`.

#### 3.1 TODO – `run_diffpir_deblur`

**What to do**:

- Mirror the **deblur branch** of `main_ddpir.py`:
  1. Load sharp HR image.
  2. Generate blurred LR observation:
     - Use `utils.utils_deblur` / `GaussialBlurOperator` / `MotionBlurOperator` similarly to the main script.
  3. Run the DiffPIR sampling loop for the deblur task.
  4. Compute PSNR / LPIPS and return an `ImageResult`.

**Call pattern**:

```python
cfg = MethodConfig(name="diffpir", task="deblur", generate_mode="DiffPIR", lambda_=..., zeta=..., sf=1)
res = run_diffpir_deblur("path/to/image.png", cfg)
```

#### 3.2 TODO – `run_dps_deblur`

Same as SR: reuse the deblur DiffPIR structure but switch to DPS data-consistency (use `grad_and_value` as in `main_ddpir.py` deblur DPS branch), controlled by `mode` (`"DPS_y0"` / `"DPS_yt"`).

#### 3.3 TODO – `run_pnp_drunet_deblur`

- Implement a PnP deblurring scheme:
  - Forward operator: convolution with blur kernel.
  - Data term: enforce `k ∗ x ≈ y`.
  - Prior term: denoiser call.
  - Return `ImageResult`.

---

### 4. Inpainting methods – `experiments/inpaint_methods.py`

Stubs:

- `run_diffpir_inpaint(img_path: str, cfg: MethodConfig) -> ImageResult`
- `run_dps_inpaint(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult`
- `run_pnp_drunet_inpaint(img_path: str, cfg: MethodConfig) -> ImageResult`

Wired into notebook through `get_inpaint_methods()`.

#### 4.1 TODO – `run_diffpir_inpaint`

**What to do**:

- Mirror the **inpaint branch** of `main_ddpir.py`:
  1. Load ground-truth image.
  2. Generate or load mask using `mask_generator` from `utils.utils_inpaint`.
  3. Apply mask to get `y = mask * x / 255`.
  4. Run the DiffPIR inpainting loop (note special handling for masked pixels).
  5. Compute PSNR / LPIPS on full image and return `ImageResult`.

#### 4.2 TODO – `run_dps_inpaint`

Same idea: inpainting but with DPS-style gradient updates, based on the inpainting DPS code path in `main_ddpir.py`.

#### 4.3 TODO – `run_pnp_drunet_inpaint`

- Implement PnP inpainting:
  - Data step: enforce that known pixels agree with the observation (`x[mask] = y[mask]` or soft constraint).
  - Prior step: denoiser call.
  - Return `ImageResult`.

---

### 5. DRUNet prior – `experiments/pnp_priors.py`

Stubs:

- `DRUNetDenoiser.__post_init__(self) -> None`
- `DRUNetDenoiser.__call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor`

There is already a working `GaussianDenoiser` that you can use while DRUNet is not ready.

#### 5.1 TODO – `DRUNetDenoiser.__post_init__`

**What to do**:

- Load:
  - DRUNet architecture.
  - Pretrained weights from `self.weights_path` (or a default).
  - Move the model to device (`self.device` or `x.device` later).

#### 5.2 TODO – `DRUNetDenoiser.__call__`

**What to do**:

- Implement:

```python
def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: (B, C, H, W)
    # 1. Optionally normalize or scale x based on DRUNet training protocol.
    # 2. Run self.model(x, sigma) or equivalent.
    # 3. Return denoised x with same shape and device.
```

Once this is done, you can plug it into the PnP runners instead of `GaussianDenoiser`.

---

### 6. Notebook + plotting – `experiments/comparison_notebook.ipynb`

Already implemented:

- `compare_task(task, testset_root, methods, sf=4) -> Dict[str, RunResult>`
  - Uses the method registries (`METHOD_REGISTRY["sr" | "deblur" | "inpaint"]`) which reference your `run_*` functions.
- `plot_metric_bar(results, metric="average_psnr", title="...")` for simple bar plots.

**Once your methods are implemented**, you can:

- For SR:

```python
sr_results = compare_task(
    task="sr",
    testset_root="testsets/demo_test",
    methods=["diffpir", "dps_y0", "dps_yt", "pnp_drunet"],
    sf=4,
)
plot_metric_bar(sr_results, metric="average_psnr", title="SR: Average PSNR")
```

- Similarly for `deblur` and `inpaint` with their respective test sets.

You can also add more plots (e.g. per-image PSNR scatter, example reconstructions) directly in the notebook.

---

### 7. Optional: extend CLI scripts with PnP

Files:

- `compare_sr_methods.py`
- `compare_deblur_methods.py`
- `compare_inpaint_methods.py`

They currently:

- Loop over methods `["diffpir", "dps_y0", "dps_yt"]`.
- Call `run_with_config` in `main_ddpir.py` (batch-style).

**Optional TODO**:

- Add support for `"pnp_drunet"` by:
  - Extending argument defaults:

    ```bash
    --methods diffpir dps_y0 dps_yt pnp_drunet
    ```

  - Adding a branch in `run_methods`:

    ```python
    elif method.lower() == "pnp_drunet":
        # Option 1: implement a batch-style PnP runner using main_ddpir infrastructure
        # Option 2: call experiments.common.run_experiment + your per-image PnP functions
    ```

This is optional because the notebook already offers a convenient comparison interface.

---

### 8. Minimal checklist

To finish the project, you can treat this as a checklist:

- **SR task**
  - [ ] Implement `run_diffpir_sr` in `experiments/sr_methods.py`.
  - [ ] Implement `run_dps_sr` in `experiments/sr_methods.py`.
  - [ ] Implement `run_pnp_drunet_sr` in `experiments/sr_methods.py`.
- **Deblur task**
  - [ ] Implement `run_diffpir_deblur` in `experiments/deblur_methods.py`.
  - [ ] Implement `run_dps_deblur` in `experiments/deblur_methods.py`.
  - [ ] Implement `run_pnp_drunet_deblur` in `experiments/deblur_methods.py`.
- **Inpaint task**
  - [ ] Implement `run_diffpir_inpaint` in `experiments/inpaint_methods.py`.
  - [ ] Implement `run_dps_inpaint` in `experiments/inpaint_methods.py`.
  - [ ] Implement `run_pnp_drunet_inpaint` in `experiments/inpaint_methods.py`.
- **DRUNet prior**
  - [ ] Implement `DRUNetDenoiser.__post_init__` in `experiments/pnp_priors.py`.
  - [ ] Implement `DRUNetDenoiser.__call__` in `experiments/pnp_priors.py`.
- **Notebook usage**
  - [ ] Run `experiments/comparison_notebook.ipynb` for SR once SR methods are implemented.
  - [ ] Run it for deblur and inpaint once those methods are implemented.

With these completed, you will have:

- DiffPIR vs DPS vs PnP+DRUNet comparisons on SR, deblur, and inpainting.
- Clean separation between:
  - per-image methods (`experiments/*_methods.py`),
  - experiment orchestration (`experiments/common.py`),
  - visualization (`experiments/comparison_notebook.ipynb`),
  - and priors (`experiments/pnp_priors.py`).

