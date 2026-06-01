#!/usr/bin/env python
# coding=utf-8
"""
Evaluate a Lotus-2 Replica disparity-normalized checkpoint on a saved split list.

The checkpoint is expected to predict normalized disparity in [0, 1]. For each
sample, this script uses the GT depth map to recover the same disparity range
used by trunc_disparity training, converts the prediction back to metric-like
relative depth, and visualizes/evaluates in depth space.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from evaluation.util import metric as depth_metric
from infer import load_lora_and_lcm_weights
from pipeline import Lotus2Pipeline
from utils.seed_all import seed_all


METRIC_NAMES = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]


CHECKPOINT_FILENAMES = {
    "core": "lotus-2_core_predictor_depth.safetensors",
    "lcm": "lotus-2_lcm_depth.safetensors",
    "detail": "lotus-2_detail_sharpener_depth.safetensors",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Lotus-2 Replica disparity checkpoint.")
    parser.add_argument("--split_file", type=str, default="replica/lotus2_train_test_split_0.5.txt")
    parser.add_argument("--replica_root", type=str, default="replica")
    parser.add_argument("--section", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/finetune_replica_disp/checkpoint-22000")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_replica_disp_checkpoint_22000")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--flux_cache_dir", type=str, default="ckpts/flux")
    parser.add_argument("--pipeline_cache_dir", type=str, default="ckpts/lotus")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--process_res", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--depth_normalization",
        type=str,
        default="trunc_disparity",
        choices=["minmax", "disparity", "trunc_disparity"],
        help="Normalization used when training this checkpoint.",
    )
    parser.add_argument("--depth_min", type=float, default=1e-5)
    parser.add_argument("--depth_max", type=float, default=None)
    parser.add_argument("--depth_trunc_quantile", type=float, default=0.02)
    parser.add_argument("--disparity_min", type=float, default=1e-6)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visualize_limit",
        type=int,
        default=-1,
        help="Maximum number of visualizations to save. Use -1 to save all samples.",
    )
    parser.add_argument("--vis_min_quantile", type=float, default=0.02)
    parser.add_argument("--vis_max_quantile", type=float, default=0.98)
    return parser.parse_args()


def get_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_split_records(split_file: Path, replica_root: Path, section_name: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    current_section: Optional[str] = None

    with split_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                continue
            if current_section is None:
                continue
            if section_name != "all" and current_section != section_name:
                continue
            if line.startswith("frame_id"):
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                raise ValueError(f"Malformed split line: {line}")
            frame_id, view_id, rgb_rel, gt_rel, pred_rel = parts[:5]
            records.append(
                {
                    "frame_id": frame_id,
                    "view_id": view_id,
                    "rgb": str(resolve_path(replica_root, rgb_rel)),
                    "gt": str(resolve_path(replica_root, gt_rel)),
                    "baseline_pred": str(resolve_path(replica_root, pred_rel)) if pred_rel else "",
                    "section": current_section,
                }
            )

    if not records:
        raise ValueError(f"No records found in section '{section_name}' of {split_file}")
    return records


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        try:
            for key in ("depth", "target", "arr_0"):
                if key in data:
                    return data[key]
            return data[data.files[0]]
        finally:
            data.close()
    return np.asarray(Image.open(path))


def squeeze_depth(array: np.ndarray) -> np.ndarray:
    depth = np.asarray(array, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            depth = depth[..., :3].mean(axis=-1)
    return depth.astype(np.float32)


def valid_depth_mask(depth: np.ndarray, depth_min: float, depth_max: Optional[float]) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max is not None:
        valid &= depth < depth_max
    return valid


def denormalize_prediction_to_depth(
    pred_01: np.ndarray,
    gt_depth: np.ndarray,
    valid: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    pred_01 = np.clip(np.asarray(pred_01, dtype=np.float32), 0.0, 1.0)
    values = gt_depth[valid]
    if values.size == 0:
        raise ValueError("GT depth has no valid pixels; cannot denormalize prediction.")

    if args.depth_normalization == "minmax":
        min_value = float(values.min())
        max_value = float(values.max())
        denom = max(max_value - min_value, 1e-6)
        pred_depth = pred_01 * denom + min_value
        pred_disparity = 1.0 / np.clip(pred_depth, args.depth_min, None)
    else:
        gt_disparity = np.zeros_like(gt_depth, dtype=np.float32)
        gt_disparity[valid] = 1.0 / np.maximum(gt_depth[valid], args.depth_min)
        disp_values = gt_disparity[valid]
        if args.depth_normalization == "trunc_disparity":
            min_value = float(np.quantile(disp_values, args.depth_trunc_quantile))
            max_value = float(np.quantile(disp_values, 1.0 - args.depth_trunc_quantile))
        else:
            min_value = float(disp_values.min())
            max_value = float(disp_values.max())
        denom = max(max_value - min_value, 1e-6)
        pred_disparity = pred_01 * denom + min_value
        pred_disparity = np.clip(pred_disparity, args.disparity_min, None)
        pred_depth = 1.0 / pred_disparity

    pred_depth = np.clip(pred_depth, args.depth_min, None)
    if args.depth_max is not None:
        pred_depth = np.clip(pred_depth, None, args.depth_max)

    return pred_depth.astype(np.float32), pred_disparity.astype(np.float32), {
        "norm_min": min_value,
        "norm_max": max_value,
    }


def maybe_resize_to_gt(array: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    if array.shape[:2] == gt_shape:
        return array.astype(np.float32)
    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((gt_shape[1], gt_shape[0]), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def colorize_depth(depth: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    depth = np.asarray(depth, dtype=np.float32)
    valid = valid & np.isfinite(depth)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalized = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    colored = (colormaps["Spectral"](1.0 - normalized)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return Image.fromarray(colored, mode="RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 24
    output = Image.new("RGB", (image.width, image.height + label_height), color=(0, 0, 0))
    output.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(output)
    draw.text((6, 5), label, fill=(255, 255, 255))
    return output


def save_visualization(
    save_path: Path,
    rgb: Image.Image,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid: np.ndarray,
    record: Dict[str, str],
    args,
):
    values = gt_depth[valid]
    if values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.quantile(values, args.vis_min_quantile))
        vmax = float(np.quantile(values, args.vis_max_quantile))
        if vmax <= vmin:
            vmin, vmax = float(values.min()), float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6

    height, width = gt_depth.shape
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.BILINEAR)

    panels = [
        add_label(rgb, f"rgb frame={record['frame_id']} view={record['view_id']}"),
        add_label(colorize_depth(gt_depth, valid, vmin, vmax), f"gt depth {vmin:.3f}-{vmax:.3f}"),
        add_label(colorize_depth(pred_depth, valid, vmin, vmax), "pred depth from disparity"),
    ]
    output = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
    x_offset = 0
    for panel in panels:
        output.paste(panel, (x_offset, 0))
        x_offset += panel.width

    save_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(save_path)


def evaluate_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    pred_ts = torch.from_numpy(pred_depth.astype(np.float32))
    gt_ts = torch.from_numpy(gt_depth.astype(np.float32))
    valid_ts = torch.from_numpy(valid.astype(bool))

    result: Dict[str, float] = {}
    for name in METRIC_NAMES:
        value = getattr(depth_metric, name)(pred_ts, gt_ts, valid_ts).item()
        result[name] = float(value)
    return result


def load_pipeline(args, device: torch.device, weight_dtype: torch.dtype) -> Lotus2Pipeline:
    checkpoint_dir = Path(args.checkpoint_dir)
    paths = {key: checkpoint_dir / filename for key, filename in CHECKPOINT_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        num_train_timesteps=10,
        cache_dir=args.flux_cache_dir,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.flux_cache_dir,
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)

    transformer, local_continuity_module = load_lora_and_lcm_weights(
        transformer,
        str(paths["core"]),
        str(paths["lcm"]),
        str(paths["detail"]),
        "depth",
    )

    pipeline = Lotus2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=scheduler,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        cache_dir=args.pipeline_cache_dir,
    )
    pipeline.local_continuity_module = local_continuity_module
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.text_encoder_2.eval()
    transformer.eval()
    local_continuity_module.eval()
    return pipeline


def choose_process_res(rgb: Image.Image, explicit_process_res: Optional[int]) -> Optional[int]:
    if explicit_process_res is not None:
        return explicit_process_res
    max_edge = max(rgb.height, rgb.width)
    if max_edge > 1024:
        return 1024
    if max_edge < 512:
        return 512
    return None


def predict_normalized_disparity(
    pipeline: Lotus2Pipeline,
    rgb: Image.Image,
    device: torch.device,
    args,
) -> np.ndarray:
    rgb_np = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    rgb_ts = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
    rgb_ts = rgb_ts / 127.5 - 1.0
    rgb_ts = rgb_ts.to(device)

    prediction = pipeline(
        rgb_in=rgb_ts,
        prompt="",
        num_inference_steps=args.num_inference_steps,
        output_type="np",
        process_res=choose_process_res(rgb, args.process_res),
    ).images[0]
    return np.clip(prediction.mean(axis=-1).astype(np.float32), 0.0, 1.0)


def write_summary(output_dir: Path, args, records: List[Dict[str, str]], averages: Dict[str, float]):
    summary_path = output_dir / "eval_metrics.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Replica disparity checkpoint evaluation\n")
        handle.write(f"split_file: {args.split_file}\n")
        handle.write(f"section: {args.section}\n")
        handle.write(f"checkpoint_dir: {args.checkpoint_dir}\n")
        handle.write(f"depth_normalization: {args.depth_normalization}\n")
        handle.write(f"samples: {len(records)}\n\n")
        for name in METRIC_NAMES:
            handle.write(f"{name}: {averages[name]:.8f}\n")


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    if not 0.0 <= args.depth_trunc_quantile < 0.5:
        raise ValueError("--depth_trunc_quantile must satisfy 0 <= q < 0.5.")
    if not 0.0 <= args.vis_min_quantile <= args.vis_max_quantile <= 1.0:
        raise ValueError("Visualization quantiles must satisfy 0 <= min <= max <= 1.")
    if args.seed is not None:
        seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_depth_dir = output_dir / "pred_depth_npy"
    pred_disp_dir = output_dir / "pred_disparity_npy"
    vis_dir = output_dir / "visualizations"
    if args.save_predictions:
        pred_depth_dir.mkdir(parents=True, exist_ok=True)
        pred_disp_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    records = load_split_records(Path(args.split_file), Path(args.replica_root), args.section)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    logging.info("Loaded %d records from %s [%s]", len(records), args.split_file, args.section)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("CUDA is not available; evaluation will be slow on CPU.")
    weight_dtype = get_weight_dtype(args.mixed_precision)
    logging.info("Loading checkpoint from %s", args.checkpoint_dir)
    pipeline = load_pipeline(args, device, weight_dtype)

    metric_sums = {name: 0.0 for name in METRIC_NAMES}
    metric_counts = {name: 0 for name in METRIC_NAMES}
    per_sample_path = output_dir / "per_sample_metrics.csv"
    fieldnames = [
        "section",
        "frame_id",
        "view_id",
        "rgb",
        "gt",
        "norm_min",
        "norm_max",
        *METRIC_NAMES,
    ]

    with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            for sample_index, record in enumerate(tqdm(records, desc="Evaluating Replica split")):
                rgb = Image.open(record["rgb"]).convert("RGB")
                gt_depth = squeeze_depth(load_array(Path(record["gt"])))
                valid = valid_depth_mask(gt_depth, args.depth_min, args.depth_max)
                if not valid.any():
                    logging.warning("Skipping %s/%s: no valid depth", record["frame_id"], record["view_id"])
                    continue

                pred_01 = predict_normalized_disparity(pipeline, rgb, device, args)
                pred_01 = maybe_resize_to_gt(pred_01, gt_depth.shape)
                pred_depth, pred_disp, norm_stats = denormalize_prediction_to_depth(pred_01, gt_depth, valid, args)
                pred_depth = maybe_resize_to_gt(pred_depth, gt_depth.shape)
                pred_disp = maybe_resize_to_gt(pred_disp, gt_depth.shape)

                safe_name = f"frame_{record['frame_id']}_view_{record['view_id']}"
                if args.save_predictions:
                    np.save(pred_depth_dir / f"{safe_name}.npy", pred_depth)
                    np.save(pred_disp_dir / f"{safe_name}.npy", pred_disp)

                if args.save_visualizations and (args.visualize_limit < 0 or sample_index < args.visualize_limit):
                    save_visualization(vis_dir / f"{safe_name}.png", rgb, gt_depth, pred_depth, valid, record, args)

                sample_metrics = evaluate_metrics(pred_depth, gt_depth, valid)
                for name, value in sample_metrics.items():
                    metric_sums[name] += value
                    metric_counts[name] += 1

                writer.writerow(
                    {
                        "section": record["section"],
                        "frame_id": record["frame_id"],
                        "view_id": record["view_id"],
                        "rgb": record["rgb"],
                        "gt": record["gt"],
                        "norm_min": norm_stats["norm_min"],
                        "norm_max": norm_stats["norm_max"],
                        **sample_metrics,
                    }
                )

    averages = {
        name: metric_sums[name] / metric_counts[name] if metric_counts[name] else float("nan")
        for name in METRIC_NAMES
    }
    write_summary(output_dir, args, records, averages)

    logging.info("Evaluation complete. Results saved to %s", output_dir)
    for name in METRIC_NAMES:
        logging.info("%s: %.8f", name, averages[name])


if __name__ == "__main__":
    main()
