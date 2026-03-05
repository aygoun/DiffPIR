import argparse

from main_ddpir import load_config_from_path, run_with_config, Config


def _clone_config(base: Config) -> Config:
    def to_dict(obj):
        if isinstance(obj, Config):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        return obj

    from copy import deepcopy

    return Config(deepcopy(to_dict(base)))


def run_methods(opt_path: str, methods):
    base_config = load_config_from_path(opt_path)

    for method in methods:
        cfg = _clone_config(base_config)
        if method.lower() == "diffpir":
            cfg.generate_mode = "DiffPIR"
        elif method.lower() == "dps_y0":
            cfg.generate_mode = "DPS_y0"
        elif method.lower() == "dps_yt":
            cfg.generate_mode = "DPS_yt"
        else:
            raise ValueError(f"Unknown method: {method}")

        cfg.result_name = (
            f"{cfg.testset_name}_{cfg.task}_{cfg.generate_mode}_"
            f"{cfg.model_name}_sigma{cfg.noise_level_img}_"
            f"NFE{cfg.iter_num}_eta{cfg.eta}_zeta{cfg.zeta}_lambda{cfg.lambda_}"
            f"_mask_type_{cfg.mask_type}"
        )

        print(f"=== Running {method} for inpainting with config: {opt_path} ===")
        run_with_config(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare inpainting methods: DiffPIR vs DPS variants."
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="configs/inpaint.yaml",
        help="Path to the base YAML config for inpainting.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["diffpir", "dps_y0", "dps_yt"],
        help="List of methods to run: diffpir, dps_y0, dps_yt",
    )
    args = parser.parse_args()

    run_methods(args.opt, args.methods)


if __name__ == "__main__":
    main()

