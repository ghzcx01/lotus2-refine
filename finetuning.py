#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune Lotus-2 on paired RGB/geometry targets.

Example:
    module load cuda/12.3.0
    export LD_LIBRARY_PATH=/home/chenxiz/.conda/envs/lotus2/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    accelerate launch finetuning.py \
        --task_name depth \
        --train_rgb_dir /path/to/rgb \
        --train_target_dir /path/to/depth \
        --output_dir outputs/finetune_depth \
        --mixed_precision bf16

The output directory contains files that can be passed back to infer.py via:
    --core_predictor_model_path
    --lcm_model_path
    --detail_sharpener_model_path
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft import get_peft_model_state_dict
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from dataloader.replica import (
    PairedGeometryDataset,
    ReplicaGeometryDataset,
    build_replica_records,
    build_training_records,
    collate_fn,
)
from infer import load_lora_and_lcm_weights
from pipeline import Lotus2Pipeline


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Lotus-2 on paired dense prediction targets.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="FLUX base model path or HuggingFace id.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--flux_cache_dir",
        type=str,
        default="ckpts/flux",
        help="Cache dir used for scheduler and transformer, matching infer.py.",
    )
    parser.add_argument(
        "--pipeline_cache_dir",
        type=str,
        default="ckpts/lotus",
        help="Cache dir used for remaining FLUX pipeline components, matching infer.py.",
    )
    parser.add_argument("--core_predictor_model_path", type=str, default=None)
    parser.add_argument("--lcm_model_path", type=str, default=None)
    parser.add_argument("--detail_sharpener_model_path", type=str, default=None)

    parser.add_argument("--task_name", type=str, default="depth", choices=["depth", "normal"])
    parser.add_argument(
        "--input_mode",
        type=str,
        default="rgb",
        choices=["rgb", "rgbd"],
        help="Use only RGB tokens, or use pred depth tokens with RGB tokens as extra context for depth refinement.",
    )
    parser.add_argument("--train_rgb_dir", type=str, default=None)
    parser.add_argument("--train_target_dir", type=str, default=None)
    parser.add_argument(
        "--train_pred_dir",
        type=str,
        default=None,
        help="Optional predicted-depth directory with filenames matching --train_rgb_dir stems; required for --input_mode rgbd outside Replica unless train_manifest provides pred.",
    )
    parser.add_argument("--train_mask_dir", type=str, default=None)
    parser.add_argument(
        "--replica_root",
        type=str,
        default=None,
        help="Optional Replica dataset root containing gt, pred, and rgb folders.",
    )
    parser.add_argument(
        "--replica_scale",
        type=str,
        default="0.5",
        help="Replica scale subfolder under gt, pred, and rgb.",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default=None,
        help=(
            "Optional JSONL/CSV manifest. Use columns/keys rgb,target and optional pred,mask,prompt. "
            "Relative paths are resolved against the manifest directory."
        ),
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--crop_mode", type=str, default="center", choices=["direct", "center", "random", "original"])
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument(
        "--depth_normalization",
        type=str,
        default="minmax",
        choices=["none", "minmax", "inverse_minmax", "log_minmax", "disparity", "trunc_disparity"],
        help="How to normalize depth targets before encoding them with the VAE.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=1e-5,
        help="Minimum valid depth value. Non-positive/near-zero depth is masked out.",
    )
    parser.add_argument(
        "--depth_trunc_quantile",
        type=float,
        default=0.02,
        help="Lower/upper quantile used by trunc_disparity depth normalization.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=None,
        help="Optional maximum valid depth value. Depth at or above this value is masked out.",
    )
    parser.add_argument(
        "--renormalize_normals",
        action="store_true",
        help="Normalize normal-map vectors after resizing/cropping.",
    )

    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for test/eval. Replica is split by frame_id.",
    )
    parser.add_argument("--eval_split_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument(
        "--eval_max_batches",
        type=int,
        default=32,
        help="Maximum eval batches per eval pass. Use 0 or negative to evaluate the full test split.",
    )
    parser.add_argument("--visualization_steps", type=int, default=50)
    parser.add_argument("--visualization_num_samples", type=int, default=1)
    parser.add_argument(
        "--visualization_inference_steps",
        type=int,
        default=4,
        help="Detail-sharpener denoising steps used only for saved visualizations.",
    )
    parser.add_argument("--loss_curve_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--lcm_learning_rate",
        type=float,
        default=None,
        help="Optional separate LR for the local continuity module when --train_lcm is enabled.",
    )
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="lotus2_finetuning")

    parser.add_argument("--train_stage", type=str, default="both", choices=["core", "detail", "both"])
    parser.add_argument("--train_lcm", action="store_true")
    parser.add_argument("--core_loss_weight", type=float, default=1.0)
    parser.add_argument("--detail_loss_weight", type=float, default=1.0)
    parser.add_argument("--timestep_core_predictor", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--min_sigma", type=float, default=0.0)
    parser.add_argument("--max_sigma", type=float, default=1.0)
    parser.add_argument(
        "--detach_core_for_detail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach the core-predicted source latent before computing the detail-sharpener loss.",
    )
    parser.add_argument(
        "--vae_latent_mode",
        type=str,
        default="sample",
        choices=["sample", "mode"],
        help="Use latent_dist.sample() or latent_dist.mode() for VAE-encoded training targets.",
    )
    parser.add_argument(
        "--upcast_trainable_params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep LoRA/LCM trainable parameters in fp32 for stability.",
    )

    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--save_all_lotus_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resolution % 16 != 0:
        raise ValueError("--resolution must be divisible by 16 for FLUX packed latents.")
    if args.input_mode == "rgbd" and args.task_name != "depth":
        raise ValueError("--input_mode rgbd is only supported for --task_name depth.")
    if args.replica_root is None and args.train_manifest is None and (args.train_rgb_dir is None or args.train_target_dir is None):
        raise ValueError("Provide --replica_root, --train_manifest, or both --train_rgb_dir and --train_target_dir.")
    if args.min_sigma < 0 or args.max_sigma > 1 or args.min_sigma >= args.max_sigma:
        raise ValueError("--min_sigma and --max_sigma must satisfy 0 <= min < max <= 1.")
    if not 0.0 <= args.depth_trunc_quantile < 0.5:
        raise ValueError("--depth_trunc_quantile must satisfy 0 <= q < 0.5.")
    if args.core_loss_weight == 0 and args.detail_loss_weight == 0:
        raise ValueError("At least one of --core_loss_weight or --detail_loss_weight must be non-zero.")
    if not 0.0 <= args.eval_split_ratio < 1.0:
        raise ValueError("--eval_split_ratio must satisfy 0 <= ratio < 1.")
    if args.eval_steps < 0 or args.visualization_steps < 0 or args.loss_curve_steps < 0:
        raise ValueError("--eval_steps, --visualization_steps, and --loss_curve_steps must be non-negative.")

    return args



def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def encode_latents(vae, images: torch.Tensor, weight_dtype: torch.dtype, mode: str) -> torch.Tensor:
    images = images.to(dtype=weight_dtype)
    latent_dist = vae.encode(images).latent_dist
    latents = latent_dist.mode() if mode == "mode" else latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    return Lotus2Pipeline._pack_latents(
        latents,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )


def unpack_latents(latents: torch.Tensor, image_height: int, image_width: int, vae_scale_factor: int) -> torch.Tensor:
    return Lotus2Pipeline._unpack_latents(latents, image_height, image_width, vae_scale_factor)


def latent_image_ids(latents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return Lotus2Pipeline._prepare_latent_image_ids(
        latents.shape[0],
        latents.shape[2] // 2,
        latents.shape[3] // 2,
        device,
        dtype,
    )


def using_rgbd(args) -> bool:
    return getattr(args, "input_mode", "rgb") == "rgbd"


def validate_rgbd_records(records: List[Dict[str, str]], args):
    if not using_rgbd(args):
        return
    missing = [record.get("rgb", f"sample_{index}") for index, record in enumerate(records) if not record.get("pred")]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            "--input_mode rgbd requires a pred path for every sample. "
            "Use --replica_root, provide pred in --train_manifest, or set --train_pred_dir. "
            f"Missing pred for {len(missing)} samples: {preview}"
        )


def require_pred_batch(batch: Dict[str, torch.Tensor], args, device: torch.device) -> Optional[torch.Tensor]:
    if not using_rgbd(args):
        return None
    if "pred" not in batch:
        raise ValueError("--input_mode rgbd requires batches to contain pred tensors.")
    return batch["pred"].to(device)


def prepare_packed_model_input(
    query_packed_latents: torch.Tensor,
    image_ids: torch.Tensor,
    args,
    rgb_context_packed_latents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    query_token_count = query_packed_latents.shape[1]
    if not using_rgbd(args):
        return query_packed_latents, image_ids, query_token_count

    if rgb_context_packed_latents is None:
        raise ValueError("--input_mode rgbd requires RGB context latents.")
    if rgb_context_packed_latents.shape != query_packed_latents.shape:
        raise ValueError(
            "RGBD query and RGB context packed latents must have matching shapes, got "
            f"{tuple(query_packed_latents.shape)} and {tuple(rgb_context_packed_latents.shape)}."
        )

    query_ids = image_ids.clone()
    rgb_ids = image_ids.clone()
    query_ids[:, 0] = 0
    rgb_ids[:, 0] = 1
    packed_latents = torch.cat([query_packed_latents, rgb_context_packed_latents], dim=1)
    packed_image_ids = torch.cat([query_ids, rgb_ids], dim=0)
    return packed_latents, packed_image_ids, query_token_count


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred.float(), target.float(), reduction="mean")

    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest") > 0.5
    mask = mask.expand(-1, pred.shape[1], -1, -1)
    if not mask.any():
        return F.mse_loss(pred.float(), target.float(), reduction="mean") * 0.0
    return F.mse_loss(pred[mask].float(), target[mask].float(), reduction="mean")


def set_active_adapter(transformer, adapter_name: str):
    module = transformer.module if hasattr(transformer, "module") else transformer
    module.set_adapter(adapter_name)




def _frame_sort_key(frame_id: str):
    return (0, int(frame_id)) if str(frame_id).isdigit() else (1, str(frame_id))


def split_records(records: List[Dict[str, str]], args) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if args.eval_split_ratio <= 0.0 or len(records) < 2:
        return records, []

    rng = random.Random(args.eval_split_seed)
    if "frame_id" in records[0]:
        frame_ids = sorted({record["frame_id"] for record in records}, key=_frame_sort_key)
        if len(frame_ids) < 2:
            return records, []
        shuffled_frame_ids = frame_ids[:]
        rng.shuffle(shuffled_frame_ids)
        eval_frame_count = max(1, int(round(len(frame_ids) * args.eval_split_ratio)))
        eval_frame_count = min(eval_frame_count, len(frame_ids) - 1)
        eval_frame_ids = set(shuffled_frame_ids[:eval_frame_count])
        train_records = [record for record in records if record["frame_id"] not in eval_frame_ids]
        eval_records = [record for record in records if record["frame_id"] in eval_frame_ids]
    else:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        eval_count = max(1, int(round(len(records) * args.eval_split_ratio)))
        eval_count = min(eval_count, len(records) - 1)
        eval_indices = set(indices[:eval_count])
        train_records = [record for index, record in enumerate(records) if index not in eval_indices]
        eval_records = [record for index, record in enumerate(records) if index in eval_indices]

    if not train_records or not eval_records:
        return records, []
    return train_records, eval_records


def write_replica_split_file(
    replica_root: Path,
    scale: str,
    train_records: List[Dict[str, str]],
    eval_records: List[Dict[str, str]],
    args,
) -> Path:
    replica_root = replica_root.resolve()
    safe_scale = str(scale).replace("/", "_")
    split_path = replica_root / f"lotus2_train_test_split_{safe_scale}.txt"

    def rel(path_value: Optional[str]) -> str:
        if not path_value:
            return ""
        path = Path(path_value)
        try:
            return str(path.resolve().relative_to(replica_root))
        except ValueError:
            return str(path)

    def write_section(handle, name: str, records_for_section: List[Dict[str, str]]):
        handle.write(f"[{name}]\n")
        handle.write("frame_id\tview_id\trgb\tgt\tpred\n")
        for record in records_for_section:
            handle.write(
                "\t".join(
                    [
                        record.get("frame_id", ""),
                        record.get("view_id", ""),
                        rel(record.get("rgb")),
                        rel(record.get("gt") or record.get("target")),
                        rel(record.get("pred")),
                    ]
                )
                + "\n"
            )
        handle.write("\n")

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        handle.write("Lotus-2 Replica train/test split\n")
        handle.write(f"scale: {scale}\n")
        handle.write(f"eval_split_ratio: {args.eval_split_ratio}\n")
        handle.write(f"eval_split_seed: {args.eval_split_seed}\n")
        handle.write(f"train_samples: {len(train_records)}\n")
        handle.write(f"test_samples: {len(eval_records)}\n")
        handle.write("split_unit: frame_id\n\n")
        write_section(handle, "train", train_records)
        write_section(handle, "test", eval_records)
    return split_path


def _decode_latents_to_images(vae, latents: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    images = vae.decode(latents.to(dtype=weight_dtype), return_dict=False)[0]
    return images.float().clamp(-1.0, 1.0)


def _tensor_rgb_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    array = array.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _tensor_to_depth01(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 3:
        depth = tensor[:3].mean(dim=0) if tensor.shape[0] > 1 else tensor[0]
    else:
        depth = tensor
    return ((depth.clamp(-1.0, 1.0) + 1.0) * 0.5).numpy()


def _colorize_depth_jet(
    depth: np.ndarray,
    mask: Optional[np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Image.Image:
    from matplotlib import colormaps

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if mask is not None:
        valid &= mask.astype(bool)

    values = depth[valid]
    if values.size == 0:
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
    else:
        vmin = float(values.min()) if vmin is None else float(vmin)
        vmax = float(values.max()) if vmax is None else float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    normalized = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    colored = (colormaps["jet"](normalized)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return Image.fromarray(colored, mode="RGB")


def _with_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 24
    output = Image.new("RGB", (image.width, image.height + label_height), color=(0, 0, 0))
    output.paste(image, (0, label_height))
    draw = ImageDraw.Draw(output)
    draw.text((6, 5), label, fill=(255, 255, 255))
    return output


def save_depth_visualizations(
    output_dir: Path,
    global_step: int,
    epoch: int,
    batch: Dict[str, torch.Tensor],
    pred_images: torch.Tensor,
    args,
):
    if args.task_name != "depth" or args.visualization_num_samples <= 0:
        return

    vis_dir = output_dir / "visualizations" / f"epoch_{epoch + 1:04d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(args.visualization_num_samples, pred_images.shape[0])

    for index in range(sample_count):
        rgb_pil = _tensor_rgb_to_pil(batch["rgb"][index])
        gt_depth = _tensor_to_depth01(batch["target"][index])
        pred_depth = _tensor_to_depth01(pred_images[index])
        mask = batch["mask"][index, 0].detach().bool().cpu().numpy() if "mask" in batch else None

        valid = np.isfinite(gt_depth)
        if mask is not None:
            valid &= mask.astype(bool)
        values = gt_depth[valid]
        if values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(values.min()), float(values.max())
            if vmax <= vmin:
                vmax = vmin + 1e-6

        gt_pil = _colorize_depth_jet(gt_depth, mask, vmin=vmin, vmax=vmax)
        pred_pil = _colorize_depth_jet(pred_depth, mask, vmin=vmin, vmax=vmax)
        panels = [
            _with_label(rgb_pil, "input image"),
        ]
        if "pred" in batch:
            input_pred_depth = _tensor_to_depth01(batch["pred"][index])
            input_pred_pil = _colorize_depth_jet(input_pred_depth, mask, vmin=vmin, vmax=vmax)
            panels.append(_with_label(input_pred_pil, "input pred depth"))
        panels.extend(
            [
                _with_label(gt_pil, f"gt depth jet vmin={vmin:.3f} vmax={vmax:.3f}"),
                _with_label(pred_pil, "output depth jet"),
            ]
        )
        panel = Image.new("RGB", (sum(image.width for image in panels), panels[0].height))
        x_offset = 0
        for image in panels:
            panel.paste(image, (x_offset, 0))
            x_offset += image.width

        if "frame_id" in batch and "view_id" in batch:
            sample_name = f"frame_{batch['frame_id'][index]}_view_{batch['view_id'][index]}"
        else:
            sample_name = f"sample_{index}"
        panel.save(vis_dir / f"step_{global_step:06d}_{sample_name}.png")


def save_training_metrics(output_dir: Path, metrics_history: List[Dict[str, Optional[float]]]):
    if not metrics_history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_metrics.csv"
    fieldnames = [
        "step",
        "train_loss",
        "train_core",
        "train_detail",
        "eval_loss",
        "eval_core",
        "eval_detail",
        "lr",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_history:
            writer.writerow({field: row.get(field) for field in fieldnames})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [row["step"] for row in metrics_history]
        train_loss = [row.get("train_loss") for row in metrics_history]
        eval_rows = [row for row in metrics_history if row.get("eval_loss") is not None]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, train_loss, label="train loss", linewidth=1.2)
        if eval_rows:
            plt.plot(
                [row["step"] for row in eval_rows],
                [row["eval_loss"] for row in eval_rows],
                label="eval loss",
                linewidth=1.6,
                marker="o",
                markersize=3,
            )
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=160)
        plt.close()
    except Exception as error:
        logger.warning(f"Could not save loss curve plot: {error}")


def _compute_eval_losses(
    batch: Dict[str, torch.Tensor],
    transformer,
    local_continuity_module,
    pipeline: Lotus2Pipeline,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    args,
    vae_scale_factor: int,
    train_core: bool,
    train_detail: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = batch["rgb"].to(accelerator.device)
    pred = require_pred_batch(batch, args, accelerator.device)
    target = batch["target"].to(accelerator.device)
    mask = batch["mask"].to(accelerator.device)
    image_height, image_width = rgb.shape[-2:]

    rgb_latents = encode_latents(pipeline.vae, rgb, weight_dtype, args.vae_latent_mode)
    pred_latents = encode_latents(pipeline.vae, pred, weight_dtype, args.vae_latent_mode) if pred is not None else None
    target_latents = encode_latents(pipeline.vae, target, weight_dtype, args.vae_latent_mode)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=batch["prompt"],
        prompt_2=None,
        device=accelerator.device,
    )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
    text_ids = text_ids.to(device=accelerator.device, dtype=weight_dtype)

    bsz = rgb_latents.shape[0]
    image_ids = latent_image_ids(rgb_latents, accelerator.device, weight_dtype)
    guidance = None
    if unwrap_model(accelerator, transformer).config.guidance_embeds:
        guidance = torch.full([bsz], args.guidance_scale, device=accelerator.device, dtype=torch.float32)

    packed_rgb_latents = pack_latents(rgb_latents)
    packed_pred_latents = pack_latents(pred_latents) if pred_latents is not None else None
    joint_attention_kwargs = {}
    core_pred_latents = None
    core_loss = torch.zeros((), device=accelerator.device)

    if train_core or train_detail:
        core_query_latents = packed_pred_latents if packed_pred_latents is not None else packed_rgb_latents
        core_input_latents, core_image_ids, core_query_token_count = prepare_packed_model_input(
            core_query_latents,
            image_ids,
            args,
            rgb_context_packed_latents=packed_rgb_latents,
        )
        set_active_adapter(transformer, "core_predictor")
        core_timestep = torch.full(
            [bsz],
            args.timestep_core_predictor / 1000.0,
            device=accelerator.device,
            dtype=core_input_latents.dtype,
        )
        core_pred_packed = transformer(
            hidden_states=core_input_latents,
            timestep=core_timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=core_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        core_pred_packed = core_pred_packed[:, :core_query_token_count]
        core_pred_latents = unpack_latents(core_pred_packed, image_height, image_width, vae_scale_factor)
        core_pred_latents = local_continuity_module(core_pred_latents)

        if train_core or args.train_lcm:
            core_loss = masked_mse(core_pred_latents, target_latents, mask)

    detail_loss = torch.zeros((), device=accelerator.device)
    if train_detail:
        source_latents = core_pred_latents.detach()
        sigmas = torch.full([bsz], 0.5, device=accelerator.device, dtype=source_latents.dtype)
        sigma_view = sigmas.view(-1, 1, 1, 1)
        detail_input_latents = sigma_view * source_latents + (1.0 - sigma_view) * target_latents
        detail_target_velocity = source_latents - target_latents

        set_active_adapter(transformer, "detail_sharpener")
        detail_input_packed = pack_latents(detail_input_latents)
        detail_input_packed, detail_image_ids, detail_query_token_count = prepare_packed_model_input(
            detail_input_packed,
            image_ids,
            args,
            rgb_context_packed_latents=packed_rgb_latents,
        )
        detail_pred_packed = transformer(
            hidden_states=detail_input_packed,
            timestep=sigmas,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=detail_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        detail_pred_packed = detail_pred_packed[:, :detail_query_token_count]
        detail_pred_velocity = unpack_latents(detail_pred_packed, image_height, image_width, vae_scale_factor)
        detail_loss = masked_mse(detail_pred_velocity, detail_target_velocity, mask)

    loss = args.core_loss_weight * core_loss + args.detail_loss_weight * detail_loss
    return loss.detach(), core_loss.detach(), detail_loss.detach()


@torch.no_grad()
def evaluate_model(
    eval_dataloader,
    transformer,
    local_continuity_module,
    pipeline: Lotus2Pipeline,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    args,
    vae_scale_factor: int,
    train_core: bool,
    train_detail: bool,
) -> Optional[Dict[str, float]]:
    if eval_dataloader is None:
        return None

    transformer_was_training = transformer.training
    lcm_was_training = local_continuity_module.training
    transformer.eval()
    local_continuity_module.eval()

    total = torch.zeros(3, device=accelerator.device, dtype=torch.float32)
    count = 0
    max_batches = args.eval_max_batches if args.eval_max_batches and args.eval_max_batches > 0 else None
    try:
        for batch_index, batch in enumerate(eval_dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            loss, core_loss, detail_loss = _compute_eval_losses(
                batch,
                transformer,
                local_continuity_module,
                pipeline,
                accelerator,
                weight_dtype,
                args,
                vae_scale_factor,
                train_core,
                train_detail,
            )
            reduced = accelerator.reduce(torch.stack([loss.float(), core_loss.float(), detail_loss.float()]), reduction="mean")
            total += reduced
            count += 1
    finally:
        transformer.train(transformer_was_training)
        local_continuity_module.train(lcm_was_training)

    if count == 0:
        return None
    averaged = (total / count).detach().cpu().tolist()
    return {"eval_loss": averaged[0], "eval_core": averaged[1], "eval_detail": averaged[2]}


@torch.no_grad()
def predict_depth_images(
    batch: Dict[str, torch.Tensor],
    transformer,
    local_continuity_module,
    pipeline: Lotus2Pipeline,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    args,
    vae_scale_factor: int,
) -> torch.Tensor:
    transformer_was_training = transformer.training
    lcm_was_training = local_continuity_module.training
    transformer.eval()
    local_continuity_module.eval()
    try:
        rgb = batch["rgb"].to(accelerator.device)
        pred = require_pred_batch(batch, args, accelerator.device)
        image_height, image_width = rgb.shape[-2:]
        rgb_latents = encode_latents(pipeline.vae, rgb, weight_dtype, args.vae_latent_mode)
        pred_latents = encode_latents(pipeline.vae, pred, weight_dtype, args.vae_latent_mode) if pred is not None else None
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=batch["prompt"],
            prompt_2=None,
            device=accelerator.device,
        )
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
        text_ids = text_ids.to(device=accelerator.device, dtype=weight_dtype)

        bsz = rgb_latents.shape[0]
        image_ids = latent_image_ids(rgb_latents, accelerator.device, weight_dtype)
        guidance = None
        if unwrap_model(accelerator, transformer).config.guidance_embeds:
            guidance = torch.full([bsz], args.guidance_scale, device=accelerator.device, dtype=torch.float32)

        packed_rgb_latents = pack_latents(rgb_latents)
        packed_pred_latents = pack_latents(pred_latents) if pred_latents is not None else None
        core_query_latents = packed_pred_latents if packed_pred_latents is not None else packed_rgb_latents
        core_input_latents, core_image_ids, core_query_token_count = prepare_packed_model_input(
            core_query_latents,
            image_ids,
            args,
            rgb_context_packed_latents=packed_rgb_latents,
        )
        set_active_adapter(transformer, "core_predictor")
        core_timestep = torch.full(
            [bsz],
            args.timestep_core_predictor / 1000.0,
            device=accelerator.device,
            dtype=core_input_latents.dtype,
        )
        latents = transformer(
            hidden_states=core_input_latents,
            timestep=core_timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=core_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        latents = latents[:, :core_query_token_count]
        latents = unpack_latents(latents, image_height, image_width, vae_scale_factor)
        latents = local_continuity_module(latents)

        if args.visualization_inference_steps > 0:
            set_active_adapter(transformer, "detail_sharpener")
            latents = pack_latents(latents)
            sigmas = np.linspace(1.0, 1.0 / args.visualization_inference_steps, args.visualization_inference_steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                pipeline.scheduler.config.base_image_seq_len,
                pipeline.scheduler.config.max_image_seq_len,
                pipeline.scheduler.config.base_shift,
                pipeline.scheduler.config.max_shift,
            )
            timesteps, _ = retrieve_timesteps(
                pipeline.scheduler,
                args.visualization_inference_steps,
                accelerator.device,
                sigmas=sigmas,
                mu=mu,
            )
            for timestep in timesteps:
                timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)
                detail_input_latents, detail_image_ids, detail_query_token_count = prepare_packed_model_input(
                    latents,
                    image_ids,
                    args,
                    rgb_context_packed_latents=packed_rgb_latents,
                )
                noise_pred = transformer(
                    hidden_states=detail_input_latents,
                    timestep=timestep_batch / 1000.0,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=detail_image_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :detail_query_token_count]
                latents = pipeline.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            latents = unpack_latents(latents, image_height, image_width, vae_scale_factor)

        return _decode_latents_to_images(pipeline.vae, latents, weight_dtype).detach().cpu()
    finally:
        transformer.train(transformer_was_training)
        local_continuity_module.train(lcm_was_training)


def configure_trainable_params(transformer, local_continuity_module, args) -> List[torch.nn.Parameter]:
    transformer.requires_grad_(False)
    local_continuity_module.requires_grad_(False)

    train_core = args.train_stage in ("core", "both")
    train_detail = args.train_stage in ("detail", "both")

    trainable_params: List[torch.nn.Parameter] = []
    for name, param in transformer.named_parameters():
        is_core = train_core and "core_predictor" in name
        is_detail = train_detail and "detail_sharpener" in name
        if is_core or is_detail:
            param.requires_grad_(True)
            trainable_params.append(param)

    if args.train_lcm:
        local_continuity_module.requires_grad_(True)
        trainable_params.extend(list(local_continuity_module.parameters()))

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters were selected. Check --train_stage and loaded adapter names.")

    if args.upcast_trainable_params:
        for param in trainable_params:
            param.data = param.data.float()

    return trainable_params


def get_optimizer_params(transformer, local_continuity_module, args):
    transformer_params = [p for p in transformer.parameters() if p.requires_grad]
    lcm_params = [p for p in local_continuity_module.parameters() if p.requires_grad]

    if args.lcm_learning_rate is not None and lcm_params:
        transformer_param_ids = {id(p) for p in lcm_params}
        transformer_params = [p for p in transformer_params if id(p) not in transformer_param_ids]
        return [
            {"params": transformer_params, "lr": args.learning_rate},
            {"params": lcm_params, "lr": args.lcm_learning_rate},
        ]

    return [{"params": transformer_params + lcm_params, "lr": args.learning_rate}]


def _clean_lora_state_dict(state_dict: Dict[str, torch.Tensor], adapter_name: str) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("base_model.model.", "")
        cleaned[key] = value.detach().cpu()

    try:
        cleaned = convert_state_dict_to_diffusers(cleaned, adapter_name=adapter_name)
    except ValueError:
        try:
            cleaned = convert_state_dict_to_diffusers(cleaned)
        except ValueError:
            pass

    return {key.replace(f".{adapter_name}.", "."): value for key, value in cleaned.items()}


def save_lotus_weights(
    output_dir: Path,
    transformer,
    local_continuity_module,
    task_name: str,
    accelerator: Accelerator,
    save_core: bool,
    save_detail: bool,
    save_lcm: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_transformer = unwrap_model(accelerator, transformer)
    unwrapped_lcm = unwrap_model(accelerator, local_continuity_module)

    if save_core:
        core_state_dict = get_peft_model_state_dict(unwrapped_transformer, adapter_name="core_predictor")
        core_state_dict = _clean_lora_state_dict(core_state_dict, "core_predictor")
        Lotus2Pipeline.save_lora_weights(
            save_directory=output_dir,
            transformer_lora_layers=core_state_dict,
            is_main_process=accelerator.is_main_process,
            weight_name=f"lotus-2_core_predictor_{task_name}.safetensors",
            safe_serialization=True,
        )

    if save_detail:
        detail_state_dict = get_peft_model_state_dict(unwrapped_transformer, adapter_name="detail_sharpener")
        detail_state_dict = _clean_lora_state_dict(detail_state_dict, "detail_sharpener")
        Lotus2Pipeline.save_lora_weights(
            save_directory=output_dir,
            transformer_lora_layers=detail_state_dict,
            is_main_process=accelerator.is_main_process,
            weight_name=f"lotus-2_detail_sharpener_{task_name}.safetensors",
            safe_serialization=True,
        )

    if save_lcm and accelerator.is_main_process:
        torch.save(
            unwrapped_lcm.state_dict(),
            output_dir / f"lotus-2_lcm_{task_name}.safetensors",
        )


def rotate_checkpoints(output_dir: Path, total_limit: Optional[int]):
    if total_limit is None:
        return
    checkpoints = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    if len(checkpoints) <= total_limit:
        return
    for checkpoint in checkpoints[: len(checkpoints) - total_limit]:
        shutil.rmtree(checkpoint)


def write_training_args(output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    args_path = output_dir / "training_args.json"
    with args_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    report_to = None if args.report_to.lower() == "none" else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=report_to,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_training_args(output_dir, args)
    accelerator.wait_for_everyone()

    if args.replica_root is not None:
        records = build_replica_records(args)
        validate_rgbd_records(records, args)
        train_records, eval_records = split_records(records, args)
        if accelerator.is_main_process:
            split_path = write_replica_split_file(Path(args.replica_root), str(args.replica_scale), train_records, eval_records, args)
            logger.info(f"Wrote Replica train/test split to {split_path}")
        train_dataset = ReplicaGeometryDataset(train_records, args)
        eval_dataset = ReplicaGeometryDataset(eval_records, args) if eval_records else None
    else:
        records = build_training_records(args)
        validate_rgbd_records(records, args)
        train_records, eval_records = split_records(records, args)
        train_dataset = PairedGeometryDataset(train_records, args)
        eval_dataset = PairedGeometryDataset(eval_records, args) if eval_records else None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.eval_batch_size or args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )

    weight_dtype = get_weight_dtype(accelerator)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
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
    transformer.to(device=accelerator.device, dtype=weight_dtype)
    transformer, local_continuity_module = load_lora_and_lcm_weights(
        transformer,
        args.core_predictor_model_path,
        args.lcm_model_path,
        args.detail_sharpener_model_path,
        args.task_name,
    )

    pipeline = Lotus2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        cache_dir=args.pipeline_cache_dir,
    )
    pipeline.local_continuity_module = local_continuity_module
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.text_encoder_2.eval()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    trainable_params = configure_trainable_params(transformer, local_continuity_module, args)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        get_optimizer_params(transformer, local_continuity_module, args),
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if eval_dataloader is not None:
        transformer, local_continuity_module, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            transformer, local_continuity_module, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        transformer, local_continuity_module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, local_continuity_module, optimizer, train_dataloader, lr_scheduler
        )

    if accelerator.is_main_process and report_to is not None:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running Lotus-2 fine-tuning *****")
    logger.info(f"  Train examples = {len(train_dataset)}")
    logger.info(f"  Eval examples = {len(eval_dataset) if eval_dataset is not None else 0}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Train stage = {args.train_stage}, train LCM = {args.train_lcm}")
    logger.info(f"  Input mode = {args.input_mode}")
    logger.info(f"  Trainable parameters = {sum(p.numel() for p in trainable_params):,}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    global_step = 0
    train_core = args.train_stage in ("core", "both")
    train_detail = args.train_stage in ("detail", "both")
    vae_scale_factor = pipeline.vae_scale_factor
    metrics_history: List[Dict[str, Optional[float]]] = []
    fixed_visualization_batch = None
    if eval_dataloader is not None:
        try:
            fixed_visualization_batch = next(iter(eval_dataloader))
        except StopIteration:
            fixed_visualization_batch = None

    for epoch in range(args.num_train_epochs):
        transformer.train()
        local_continuity_module.train(args.train_lcm)

        for batch in train_dataloader:
            rgb = batch["rgb"].to(accelerator.device)
            pred = require_pred_batch(batch, args, accelerator.device)
            target = batch["target"].to(accelerator.device)
            mask = batch["mask"].to(accelerator.device)
            image_height, image_width = rgb.shape[-2:]

            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    rgb_latents = encode_latents(pipeline.vae, rgb, weight_dtype, args.vae_latent_mode)
                    pred_latents = (
                        encode_latents(pipeline.vae, pred, weight_dtype, args.vae_latent_mode)
                        if pred is not None
                        else None
                    )
                    target_latents = encode_latents(pipeline.vae, target, weight_dtype, args.vae_latent_mode)
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                        prompt=batch["prompt"],
                        prompt_2=None,
                        device=accelerator.device,
                    )
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                    text_ids = text_ids.to(device=accelerator.device, dtype=weight_dtype)

                bsz = rgb_latents.shape[0]
                image_ids = latent_image_ids(rgb_latents, accelerator.device, weight_dtype)
                guidance = None
                if unwrap_model(accelerator, transformer).config.guidance_embeds:
                    guidance = torch.full([bsz], args.guidance_scale, device=accelerator.device, dtype=torch.float32)

                packed_rgb_latents = pack_latents(rgb_latents)
                packed_pred_latents = pack_latents(pred_latents) if pred_latents is not None else None
                joint_attention_kwargs = {}
                core_pred_latents = None
                core_loss = torch.zeros((), device=accelerator.device)

                if train_core or train_detail:
                    core_query_latents = packed_pred_latents if packed_pred_latents is not None else packed_rgb_latents
                    core_input_latents, core_image_ids, core_query_token_count = prepare_packed_model_input(
                        core_query_latents,
                        image_ids,
                        args,
                        rgb_context_packed_latents=packed_rgb_latents,
                    )
                    core_context = torch.enable_grad() if train_core or args.train_lcm else torch.no_grad()
                    with core_context:
                        set_active_adapter(transformer, "core_predictor")
                        core_timestep = torch.full(
                            [bsz],
                            args.timestep_core_predictor / 1000.0,
                            device=accelerator.device,
                            dtype=core_input_latents.dtype,
                        )
                        core_pred_packed = transformer(
                            hidden_states=core_input_latents,
                            timestep=core_timestep,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=core_image_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        core_pred_packed = core_pred_packed[:, :core_query_token_count]
                        core_pred_latents = unpack_latents(
                            core_pred_packed,
                            image_height,
                            image_width,
                            vae_scale_factor,
                        )
                        core_pred_latents = local_continuity_module(core_pred_latents)

                    if train_core or args.train_lcm:
                        core_loss = masked_mse(core_pred_latents, target_latents, mask)

                detail_loss = torch.zeros((), device=accelerator.device)
                if train_detail:
                    if core_pred_latents is None:
                        raise RuntimeError("Detail training requires core source latents.")
                    source_latents = core_pred_latents.detach() if args.detach_core_for_detail else core_pred_latents
                    sigmas = torch.rand(bsz, device=accelerator.device, dtype=source_latents.dtype)
                    sigmas = sigmas * (args.max_sigma - args.min_sigma) + args.min_sigma
                    sigma_view = sigmas.view(-1, 1, 1, 1)
                    detail_input_latents = sigma_view * source_latents + (1.0 - sigma_view) * target_latents
                    detail_target_velocity = source_latents - target_latents

                    set_active_adapter(transformer, "detail_sharpener")
                    detail_input_packed = pack_latents(detail_input_latents)
                    detail_input_packed, detail_image_ids, detail_query_token_count = prepare_packed_model_input(
                        detail_input_packed,
                        image_ids,
                        args,
                        rgb_context_packed_latents=packed_rgb_latents,
                    )
                    detail_pred_packed = transformer(
                        hidden_states=detail_input_packed,
                        timestep=sigmas,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=detail_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    detail_pred_packed = detail_pred_packed[:, :detail_query_token_count]
                    detail_pred_velocity = unpack_latents(
                        detail_pred_packed,
                        image_height,
                        image_width,
                        vae_scale_factor,
                    )
                    detail_loss = masked_mse(detail_pred_velocity, detail_target_velocity, mask)

                loss = args.core_loss_weight * core_loss + args.detail_loss_weight * detail_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in list(transformer.parameters()) + list(local_continuity_module.parameters()) if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "core": core_loss.detach().item(),
                    "detail": detail_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                metric_row = {
                    "step": global_step,
                    "train_loss": logs["loss"],
                    "train_core": logs["core"],
                    "train_detail": logs["detail"],
                    "eval_loss": None,
                    "eval_core": None,
                    "eval_detail": None,
                    "lr": logs["lr"],
                }

                if eval_dataloader is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_metrics = evaluate_model(
                        eval_dataloader,
                        transformer,
                        local_continuity_module,
                        pipeline,
                        accelerator,
                        weight_dtype,
                        args,
                        vae_scale_factor,
                        train_core,
                        train_detail,
                    )
                    if eval_metrics is not None:
                        logs.update(eval_metrics)
                        metric_row.update(eval_metrics)

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    metrics_history.append(metric_row)

                    if args.visualization_steps > 0 and global_step % args.visualization_steps == 0:
                        visualization_batch = fixed_visualization_batch if fixed_visualization_batch is not None else batch
                        pred_images = predict_depth_images(
                            visualization_batch,
                            unwrap_model(accelerator, transformer),
                            unwrap_model(accelerator, local_continuity_module),
                            pipeline,
                            accelerator,
                            weight_dtype,
                            args,
                            vae_scale_factor,
                        )
                        save_depth_visualizations(output_dir, global_step, epoch, visualization_batch, pred_images, args)

                    if args.loss_curve_steps > 0 and global_step % args.loss_curve_steps == 0:
                        save_training_metrics(output_dir, metrics_history)

                    if global_step % args.checkpointing_steps == 0:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        rotate_checkpoints(output_dir, args.checkpoints_total_limit)
                        save_lotus_weights(
                            checkpoint_dir,
                            transformer,
                            local_continuity_module,
                            args.task_name,
                            accelerator,
                            save_core=args.save_all_lotus_weights or train_core,
                            save_detail=args.save_all_lotus_weights or train_detail,
                            save_lcm=args.save_all_lotus_weights or args.train_lcm,
                        )
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    final_eval_metrics = None
    if eval_dataloader is not None and args.eval_steps > 0:
        final_eval_metrics = evaluate_model(
            eval_dataloader,
            transformer,
            local_continuity_module,
            pipeline,
            accelerator,
            weight_dtype,
            args,
            vae_scale_factor,
            train_core,
            train_detail,
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if final_eval_metrics is not None:
            logger.info(
                "Final eval loss: "
                f"{final_eval_metrics['eval_loss']:.6f} "
                f"core={final_eval_metrics['eval_core']:.6f} "
                f"detail={final_eval_metrics['eval_detail']:.6f}"
            )
            if metrics_history and metrics_history[-1]["step"] == global_step:
                metrics_history[-1].update(final_eval_metrics)
            else:
                metrics_history.append({"step": global_step, **final_eval_metrics})
        save_training_metrics(output_dir, metrics_history)
        save_lotus_weights(
            output_dir,
            transformer,
            local_continuity_module,
            args.task_name,
            accelerator,
            save_core=args.save_all_lotus_weights or train_core,
            save_detail=args.save_all_lotus_weights or train_detail,
            save_lcm=args.save_all_lotus_weights or args.train_lcm,
        )
        logger.info(f"Saved final Lotus-2 fine-tuned weights to {output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
