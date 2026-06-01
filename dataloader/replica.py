import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_EXTENSIONS = IMAGE_EXTENSIONS | {".npy", ".npz"}
REPLICA_VIEW_TO_STITCHED_INDEX = {0: 0, 3: 1}


def _list_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions])


def _resolve_manifest_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    base_dir = manifest_path.parent
    if manifest_path.suffix.lower() == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "rgb" not in item or "target" not in item:
                    raise ValueError(f"Manifest line {line_number} must contain rgb and target.")
                records.append(
                    {
                        "rgb": _resolve_manifest_path(item["rgb"], base_dir),
                        "target": _resolve_manifest_path(item["target"], base_dir),
                        "pred": _resolve_manifest_path(item["pred"], base_dir) if item.get("pred") else None,
                        "mask": _resolve_manifest_path(item["mask"], base_dir) if item.get("mask") else None,
                        "prompt": item.get("prompt", ""),
                    }
                )
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                if not row.get("rgb") or not row.get("target"):
                    raise ValueError(f"Manifest row {row_number} must contain rgb and target.")
                records.append(
                    {
                        "rgb": _resolve_manifest_path(row["rgb"], base_dir),
                        "target": _resolve_manifest_path(row["target"], base_dir),
                        "pred": _resolve_manifest_path(row["pred"], base_dir) if row.get("pred") else None,
                        "mask": _resolve_manifest_path(row["mask"], base_dir) if row.get("mask") else None,
                        "prompt": row.get("prompt", ""),
                    }
                )
    else:
        raise ValueError("--train_manifest must be a .jsonl or .csv file.")

    return records


def build_training_records(args) -> List[Dict[str, str]]:
    if args.train_manifest is not None:
        records = _read_manifest(Path(args.train_manifest))
    else:
        rgb_dir = Path(args.train_rgb_dir)
        target_dir = Path(args.train_target_dir)
        mask_dir = Path(args.train_mask_dir) if args.train_mask_dir else None
        pred_dir = Path(args.train_pred_dir) if getattr(args, "train_pred_dir", None) else None

        target_by_stem = {p.stem: p for p in _list_files(target_dir, TARGET_EXTENSIONS)}
        mask_by_stem = {p.stem: p for p in _list_files(mask_dir, IMAGE_EXTENSIONS)} if mask_dir else {}
        pred_by_stem = {p.stem: p for p in _list_files(pred_dir, TARGET_EXTENSIONS)} if pred_dir else {}

        records = []
        missing = []
        missing_pred = []
        for rgb_path in _list_files(rgb_dir, IMAGE_EXTENSIONS):
            target_path = target_by_stem.get(rgb_path.stem)
            if target_path is None:
                missing.append(rgb_path.name)
                continue
            pred_path = pred_by_stem.get(rgb_path.stem) if pred_dir else None
            if pred_dir and pred_path is None:
                missing_pred.append(rgb_path.name)
                continue
            records.append(
                {
                    "rgb": str(rgb_path),
                    "target": str(target_path),
                    "pred": str(pred_path) if pred_path is not None else None,
                    "mask": str(mask_by_stem[rgb_path.stem]) if rgb_path.stem in mask_by_stem else None,
                    "prompt": "",
                }
            )

        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"Missing target files with matching stems for {len(missing)} RGB images: {preview}")
        if missing_pred:
            preview = ", ".join(missing_pred[:10])
            raise ValueError(f"Missing pred files with matching stems for {len(missing_pred)} RGB images: {preview}")

    if args.max_train_samples is not None:
        records = records[: args.max_train_samples]
    if len(records) == 0:
        raise ValueError("No training pairs found.")
    return records


def _parse_replica_rgb_stem(stem: str) -> Optional[Tuple[str, int]]:
    if not stem.startswith("rgb_"):
        return None
    frame_id, separator, view_id = stem[len("rgb_") :].rpartition("_")
    if not separator:
        return None
    try:
        return frame_id, int(view_id)
    except ValueError:
        return None


def _record_sort_key(record: Dict[str, str]):
    frame_id = record.get("frame_id", "")
    frame_key = (0, int(frame_id)) if frame_id.isdigit() else (1, frame_id)
    return frame_key, int(record.get("view_id", 0))


def build_replica_records(args) -> List[Dict[str, str]]:
    replica_root = Path(args.replica_root)
    scale = str(args.replica_scale)
    rgb_dir = replica_root / "rgb" / scale
    gt_dir = replica_root / "gt" / scale
    pred_dir = replica_root / "pred" / scale

    for directory in (rgb_dir, gt_dir, pred_dir):
        if not directory.exists():
            raise ValueError(f"Replica directory does not exist: {directory}")

    records: List[Dict[str, str]] = []
    missing: List[str] = []
    for rgb_path in _list_files(rgb_dir, IMAGE_EXTENSIONS):
        parsed = _parse_replica_rgb_stem(rgb_path.stem)
        if parsed is None:
            continue

        frame_id, view_id = parsed
        if view_id not in REPLICA_VIEW_TO_STITCHED_INDEX:
            continue

        gt_path = gt_dir / f"depth_{frame_id}_{view_id}.npy"
        stitched_index = REPLICA_VIEW_TO_STITCHED_INDEX[view_id]
        pred_path = pred_dir / frame_id / f"depth_stitched_{stitched_index}.npy"
        missing_paths = [str(path) for path in (gt_path, pred_path) if not path.exists()]
        if missing_paths:
            missing.append(f"{rgb_path.name}: {', '.join(missing_paths)}")
            continue

        records.append(
            {
                "rgb": str(rgb_path),
                "gt": str(gt_path),
                "pred": str(pred_path),
                "target": str(gt_path),
                "mask": None,
                "prompt": "",
                "frame_id": frame_id,
                "view_id": str(view_id),
            }
        )

    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"Missing Replica gt/pred files for {len(missing)} RGB images:\n{preview}")

    records = sorted(records, key=_record_sort_key)
    if args.max_train_samples is not None:
        records = records[: args.max_train_samples]
    if len(records) == 0:
        raise ValueError("No Replica training samples found.")
    return records


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        try:
            for key in ("depth", "normal", "target", "arr_0"):
                if key in data:
                    return data[key]
            return data[data.files[0]]
        finally:
            data.close()
    return np.array(Image.open(path))


def _as_hwc(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    return array


def _depth_to_01(
    depth: np.ndarray,
    normalization: str,
    min_depth: float = 1e-5,
    max_depth: Optional[float] = None,
    trunc_quantile: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0] if depth.shape[-1] == 1 else depth[..., :3].mean(axis=-1)

    valid = np.isfinite(depth) & (depth > min_depth)
    if max_depth is not None:
        valid &= depth < max_depth
    clean = np.where(valid, depth, 0.0).astype(np.float32)

    if normalization == "none":
        depth_01 = clean
        if depth_01.max(initial=0.0) > 1.0 or depth_01.min(initial=0.0) < 0.0:
            depth_01 = np.clip(depth_01, 0.0, 1.0)
    else:
        values = clean[valid]
        if values.size == 0:
            depth_01 = np.zeros_like(clean, dtype=np.float32)
        else:
            if normalization == "log_minmax":
                eps = 1e-6
                work = np.log(np.maximum(clean, eps))
                values = work[valid]
            elif normalization in ("disparity", "trunc_disparity"):
                work = np.zeros_like(clean, dtype=np.float32)
                work[valid] = 1.0 / np.maximum(clean[valid], min_depth)
                values = work[valid]
            else:
                work = clean

            if normalization == "trunc_disparity":
                min_value = float(np.quantile(values, trunc_quantile))
                max_value = float(np.quantile(values, 1.0 - trunc_quantile))
            else:
                min_value = float(values.min())
                max_value = float(values.max())
            denom = max(max_value - min_value, 1e-6)
            depth_01 = (work - min_value) / denom
            depth_01 = np.clip(depth_01, 0.0, 1.0)
            if normalization == "inverse_minmax":
                depth_01 = 1.0 - depth_01

    depth_01[~valid] = 0.0
    return depth_01.astype(np.float32), valid


def _normal_to_minus1_1(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    array = _as_hwc(array).astype(np.float32)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Normal target must have 3 channels, got shape {array.shape}.")

    valid = np.isfinite(array).all(axis=-1)
    clean = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = float(clean.min(initial=0.0))
    max_value = float(clean.max(initial=0.0))
    if min_value >= -1.0 and max_value <= 1.0:
        normal = clean
        if min_value >= 0.0:
            normal = normal * 2.0 - 1.0
    else:
        normal = clean / 127.5 - 1.0

    return np.clip(normal, -1.0, 1.0).astype(np.float32), valid


def _to_tensor_chw(array: np.ndarray) -> torch.Tensor:
    array = _as_hwc(array)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resize_tensor(tensor: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    tensor = tensor.unsqueeze(0)
    if mode == "nearest":
        tensor = F.interpolate(tensor, size=size, mode=mode)
    else:
        tensor = F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return tensor.squeeze(0)


def _resize_short_side(tensor: torch.Tensor, resolution: int, mode: str) -> torch.Tensor:
    _, height, width = tensor.shape
    scale = resolution / min(height, width)
    new_height = max(resolution, int(round(height * scale)))
    new_width = max(resolution, int(round(width * scale)))
    return _resize_tensor(tensor, (new_height, new_width), mode)


def _crop_tensor(tensor: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    return tensor[:, top : top + size, left : left + size]


class _GeometryDatasetBase(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict[str, str]], args):
        self.records = records
        self.args = args

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_rgb(path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        return _to_tensor_chw(image_np)

    def _load_target(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        array = _load_array(Path(path))
        if self.args.task_name == "depth":
            depth_01, valid = _depth_to_01(
                array,
                self.args.depth_normalization,
                min_depth=getattr(self.args, "depth_min", 1e-5),
                max_depth=getattr(self.args, "depth_max", None),
                trunc_quantile=getattr(self.args, "depth_trunc_quantile", 0.02),
            )
            target = np.repeat(depth_01[..., None] * 2.0 - 1.0, 3, axis=-1)
        else:
            target, valid = _normal_to_minus1_1(array)

        target_tensor = _to_tensor_chw(target.astype(np.float32))
        mask_tensor = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
        return target_tensor, mask_tensor

    @staticmethod
    def _load_mask(path: Optional[str]) -> Optional[torch.Tensor]:
        if path is None:
            return None
        mask = Image.open(path).convert("L")
        mask_np = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        return torch.from_numpy(mask_np).unsqueeze(0)

    def _prepare_tensors(
        self,
        rgb: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        extra_targets: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        extra_targets = extra_targets or []

        rgb_height, rgb_width = rgb.shape[-2:]
        if target.shape[-2:] != (rgb_height, rgb_width):
            target = _resize_tensor(target, (rgb_height, rgb_width), "bilinear")
            target_mask = _resize_tensor(target_mask, (rgb_height, rgb_width), "nearest")
        extra_targets = [
            _resize_tensor(extra, (rgb_height, rgb_width), "bilinear")
            if extra.shape[-2:] != (rgb_height, rgb_width)
            else extra
            for extra in extra_targets
        ]

        if self.args.crop_mode == "original":
            pass
        elif self.args.crop_mode == "direct":
            rgb = _resize_tensor(rgb, (self.args.resolution, self.args.resolution), "bilinear")
            target = _resize_tensor(target, (self.args.resolution, self.args.resolution), "bilinear")
            target_mask = _resize_tensor(target_mask, (self.args.resolution, self.args.resolution), "nearest")
            extra_targets = [
                _resize_tensor(extra, (self.args.resolution, self.args.resolution), "bilinear")
                for extra in extra_targets
            ]
        else:
            rgb = _resize_short_side(rgb, self.args.resolution, "bilinear")
            target = _resize_short_side(target, self.args.resolution, "bilinear")
            target_mask = _resize_short_side(target_mask, self.args.resolution, "nearest")
            extra_targets = [
                _resize_short_side(extra, self.args.resolution, "bilinear") for extra in extra_targets
            ]

            _, height, width = rgb.shape
            if self.args.crop_mode == "random":
                top = random.randint(0, height - self.args.resolution)
                left = random.randint(0, width - self.args.resolution)
            else:
                top = (height - self.args.resolution) // 2
                left = (width - self.args.resolution) // 2
            rgb = _crop_tensor(rgb, top, left, self.args.resolution)
            target = _crop_tensor(target, top, left, self.args.resolution)
            target_mask = _crop_tensor(target_mask, top, left, self.args.resolution)
            extra_targets = [_crop_tensor(extra, top, left, self.args.resolution) for extra in extra_targets]

        _, final_height, final_width = rgb.shape
        if final_height % 16 != 0 or final_width % 16 != 0:
            raise ValueError(
                f"Final training size {(final_height, final_width)} must be divisible by 16. "
                "Use a different resize/crop mode or preprocess the inputs."
            )

        did_flip = self.args.random_flip and random.random() < 0.5
        if did_flip:
            rgb = torch.flip(rgb, dims=[2])
            target = torch.flip(target, dims=[2])
            target_mask = torch.flip(target_mask, dims=[2])
            extra_targets = [torch.flip(extra, dims=[2]) for extra in extra_targets]
            if self.args.task_name == "normal":
                target[0] = -target[0]
                for extra in extra_targets:
                    extra[0] = -extra[0]

        if self.args.task_name == "normal" and self.args.renormalize_normals:
            target = F.normalize(target, dim=0, eps=1e-6).clamp(-1.0, 1.0)
            extra_targets = [F.normalize(extra, dim=0, eps=1e-6).clamp(-1.0, 1.0) for extra in extra_targets]

        rgb = rgb * 2.0 - 1.0
        return rgb, target, target_mask, extra_targets


class PairedGeometryDataset(_GeometryDatasetBase):
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        rgb = self._load_rgb(record["rgb"])
        target, target_mask = self._load_target(record["target"])
        explicit_mask = self._load_mask(record.get("mask"))
        if explicit_mask is not None:
            if explicit_mask.shape[-2:] != target_mask.shape[-2:]:
                explicit_mask = _resize_tensor(explicit_mask, target_mask.shape[-2:], "nearest")
            target_mask = target_mask * explicit_mask

        pred = None
        if record.get("pred"):
            pred, pred_mask = self._load_target(record["pred"])
            if pred_mask.shape[-2:] != target_mask.shape[-2:]:
                pred_mask = _resize_tensor(pred_mask, target_mask.shape[-2:], "nearest")
            target_mask = target_mask * pred_mask

        rgb, target, target_mask, extra_targets = self._prepare_tensors(
            rgb,
            target,
            target_mask,
            [pred] if pred is not None else None,
        )
        result = {
            "rgb": rgb.contiguous(),
            "target": target.contiguous(),
            "mask": (target_mask > 0.5).contiguous(),
            "prompt": record.get("prompt", ""),
        }
        if extra_targets:
            result["pred"] = extra_targets[0].contiguous()
        return result


class ReplicaGeometryDataset(_GeometryDatasetBase):
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        rgb = self._load_rgb(record["rgb"])
        gt, gt_mask = self._load_target(record["gt"])
        pred, pred_mask = self._load_target(record["pred"])
        if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
            pred_mask = _resize_tensor(pred_mask, gt_mask.shape[-2:], "nearest")
        target_mask = gt_mask * pred_mask



        rgb, gt, target_mask, extra_targets = self._prepare_tensors(rgb, gt, target_mask, [pred])
        pred = extra_targets[0]
        return {
            "rgb": rgb.contiguous(),
            "gt": gt.contiguous(),
            "pred": pred.contiguous(),
            "target": gt.contiguous(),
            "mask": (target_mask > 0.5).contiguous(),
            "prompt": record.get("prompt", ""),
            "frame_id": record["frame_id"],
            "view_id": record["view_id"],
        }


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {
        "rgb": torch.stack([example["rgb"] for example in examples]),
        "target": torch.stack([example["target"] for example in examples]),
        "mask": torch.stack([example["mask"] for example in examples]),
        "prompt": [example["prompt"] for example in examples],
    }
    if "gt" in examples[0]:
        batch["gt"] = torch.stack([example["gt"] for example in examples])
    if "pred" in examples[0]:
        batch["pred"] = torch.stack([example["pred"] for example in examples])
    if "frame_id" in examples[0]:
        batch["frame_id"] = [example["frame_id"] for example in examples]
    if "view_id" in examples[0]:
        batch["view_id"] = [example["view_id"] for example in examples]
    return batch
