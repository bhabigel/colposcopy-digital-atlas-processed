# Preprocessing utilities for the AnnoCerv segmentation workflow.

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Keep dataset discovery and export usable even when PyTorch is not installed.
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_AVAILABLE = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

ANNOTATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "squamocolumnar_junction": (0, 0, 255),
    "acetowhite_area": (128, 0, 128),
    "atypical_vessels_punctation": (255, 0, 0),
    "mosaic": (165, 42, 42),
    "naboth_cyst": (255, 255, 0),
    "cuffed_gland_opening": (0, 0, 0),
}

CLASS_TO_INDEX = {name: idx + 1 for idx, name in enumerate(ANNOTATION_COLORS)}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}


@dataclass(frozen=True)
class SplitConfig:

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        if min(self.train_size, self.val_size, self.test_size) <= 0:
            raise ValueError("All split proportions must be positive")


def seed_everything(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_dataset_root(dataset_root: Optional[str | Path] = None) -> Path:

    candidates: List[Path] = []
    if dataset_root is not None:
        candidates.append(Path(dataset_root))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "AnnoCerv" / "dataset",
            cwd / "dataset",
            cwd / "annocerv" / "dataset",
            Path("/kaggle/input/annocerv/dataset"),
            Path("/kaggle/input/annocerv/AnnoCerv/dataset"),
            Path("/kaggle/input/annocerv"),
            Path("/kaggle/working/AnnoCerv/dataset"),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            has_cases = any(p.is_dir() and p.name.lower().startswith("case") for p in candidate.iterdir())
            has_scores = (candidate / "swede_scores.csv").exists()
            if has_cases or has_scores:
                return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"AnnoCerv dataset directory was not found. Searched:\n{searched}")


def resolve_processed_dataset_root(processed_root: Optional[str | Path] = None, required: bool = True) -> Optional[Path]:

    candidates: List[Path] = []
    if processed_root is not None:
        candidates.append(Path(processed_root))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "processed_annocerv",
            cwd / "processed-annocerv",
            Path("/kaggle/input/processed-annocerv/processed_annocerv"),
            Path("/kaggle/input/processed-annocerv"),
            Path("/kaggle/input/processed_annocerv"),
            Path("/kaggle/working/processed_annocerv"),
        ]
    )

    for candidate in candidates:
        metadata = candidate / "metadata.csv"
        has_split_dirs = all((candidate / split / "images").exists() for split in ["train", "validation", "test"])
        if candidate.exists() and candidate.is_dir() and (metadata.exists() or has_split_dirs):
            return candidate.resolve()

    if required:
        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Processed AnnoCerv dataset was not found. Searched:\n{searched}")
    return None


def parse_annocerv_filename(path: str | Path) -> Dict[str, object]:

    name = Path(path).stem
    match = re.match(r"^C(?P<case_id>\d+)(?P<image_type>[A-Za-z]+)\s*\((?P<image_index>\d+)\)$", name)
    if not match:
        return {
            "case_id": None,
            "case_label": None,
            "image_type": "unknown",
            "image_index": None,
            "image_id": name,
        }

    case_id = int(match.group("case_id"))
    image_type = match.group("image_type")
    image_index = int(match.group("image_index"))
    return {
        "case_id": case_id,
        "case_label": f"Case {case_id}",
        "image_type": image_type,
        "image_index": image_index,
        "image_id": f"C{case_id}{image_type}_{image_index}",
    }


def load_swede_scores(dataset_root: Optional[str | Path] = None) -> pd.DataFrame:

    root = resolve_dataset_root(dataset_root)
    path = root / "swede_scores.csv"
    if not path.exists():
        return pd.DataFrame(columns=["case_id", "swede_score"])

    raw = pd.read_csv(path, header=None)
    if raw.empty:
        return pd.DataFrame(columns=["case_id", "swede_score"])

    if raw.shape[1] == 1:
        scores = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        return pd.DataFrame({"case_id": np.arange(1, len(scores) + 1), "swede_score": scores})

    lower_cols = [str(col).lower() for col in raw.iloc[0].tolist()]
    if any("swede" in col for col in lower_cols):
        raw = pd.read_csv(path)

    columns = {str(col).lower(): col for col in raw.columns}
    case_col = next((columns[col] for col in columns if "case" in col or col in {"id", "case_id"}), raw.columns[0])
    score_col = next((columns[col] for col in columns if "swede" in col or "score" in col), raw.columns[-1])

    out = raw[[case_col, score_col]].copy()
    out.columns = ["case_id", "swede_score"]
    out["case_id"] = pd.to_numeric(out["case_id"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    out["swede_score"] = pd.to_numeric(out["swede_score"], errors="coerce")
    return out.dropna(subset=["case_id"]).astype({"case_id": int})


def collect_annocerv_records(
    dataset_root: Optional[str | Path] = None,
    image_type: str = "Aceto",
    require_mask: bool = True,
) -> pd.DataFrame:

    root = resolve_dataset_root(dataset_root)
    swede = load_swede_scores(root)
    swede_map = dict(zip(swede["case_id"], swede["swede_score"])) if not swede.empty else {}

    records: List[Dict[str, object]] = []
    for case_dir in sorted(root.glob("Case *"), key=lambda p: int(re.search(r"\d+", p.name).group(0)) if re.search(r"\d+", p.name) else 0):
        if not case_dir.is_dir():
            continue
        for image_path in sorted(case_dir.glob("*.jpg")):
            parsed = parse_annocerv_filename(image_path)
            if str(parsed["image_type"]).lower() != image_type.lower():
                continue

            mask_path = image_path.with_suffix(".png")
            if require_mask and not mask_path.exists():
                continue

            width, height = Image.open(image_path).size
            has_mask = mask_path.exists()
            mask_area = np.nan
            if has_mask:
                mask_area = int(read_annotation_mask(mask_path, image_size=None, binary=True).sum())

            record = {
                **parsed,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if has_mask else None,
                "case_dir": str(case_dir),
                "original_width": width,
                "original_height": height,
                "has_mask": has_mask,
                "mask_area_original": mask_area,
                "has_positive_mask": bool(mask_area > 0) if not pd.isna(mask_area) else False,
                "swede_score": swede_map.get(parsed["case_id"], np.nan),
            }
            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No {image_type} images were found under {root}")
    return df.sort_values(["case_id", "image_index"]).reset_index(drop=True)


def read_rgb_image(path: str | Path, image_size: Optional[Tuple[int, int]] = DEFAULT_IMAGE_SIZE) -> np.ndarray:

    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def read_annotation_mask(
    path: str | Path,
    image_size: Optional[Tuple[int, int]] = DEFAULT_IMAGE_SIZE,
    binary: bool = True,
) -> np.ndarray:

    rgba = Image.open(path).convert("RGBA")
    if image_size is not None:
        # Nearest-neighbour resizing preserves discrete mask labels.
        rgba = rgba.resize((image_size[1], image_size[0]), Image.NEAREST)
    arr = np.asarray(rgba)
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    visible = alpha > 0
    non_white = np.any(rgb < 250, axis=-1)
    foreground = visible & non_white

    if binary:
        return foreground.astype(np.float32)

    label = np.zeros(foreground.shape, dtype=np.uint8)
    rgb_int = rgb.astype(np.int16)
    # A small colour tolerance handles compression and export artefacts.
    for class_name, class_index in CLASS_TO_INDEX.items():
        color = np.asarray(ANNOTATION_COLORS[class_name], dtype=np.int16)
        distance = np.linalg.norm(rgb_int - color, axis=-1)
        label[(distance <= 60) & foreground] = class_index
    label[(label == 0) & foreground] = CLASS_TO_INDEX["acetowhite_area"]
    return label


def read_binary_mask_image(path: str | Path, image_size: Optional[Tuple[int, int]] = DEFAULT_IMAGE_SIZE) -> np.ndarray:

    mask = Image.open(path).convert("L")
    if image_size is not None:
        mask = mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)


def read_class_mask_image(path: str | Path, image_size: Optional[Tuple[int, int]] = DEFAULT_IMAGE_SIZE) -> np.ndarray:

    mask = Image.open(path).convert("L")
    if image_size is not None:
        mask = mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def normalize_image(image: np.ndarray, mean: Sequence[float] = DEFAULT_MEAN, std: Sequence[float] = DEFAULT_STD) -> np.ndarray:

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return (image - mean_arr) / std_arr


def split_records(
    records: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    group_col: str = "case_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    config.validate()
    rng = np.random.default_rng(config.random_state)
    # Shuffle case IDs, not individual images, to prevent patient-level leakage.
    groups = np.array(sorted(records[group_col].dropna().unique()))
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = max(1, int(round(n_groups * config.train_size)))
    n_val = max(1, int(round(n_groups * config.val_size)))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    def subset(group_set: set, name: str) -> pd.DataFrame:
        out = records[records[group_col].isin(group_set)].copy().reset_index(drop=True)
        out["split"] = name
        return out

    return subset(train_groups, "train"), subset(val_groups, "validation"), subset(test_groups, "test")


def prepare_splits(
    dataset_root: Optional[str | Path] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    split_config: SplitConfig = SplitConfig(),
) -> Dict[str, pd.DataFrame]:

    records = collect_annocerv_records(dataset_root)
    train_df, val_df, test_df = split_records(records, split_config)
    for frame in (train_df, val_df, test_df):
        frame["target_height"] = image_size[0]
        frame["target_width"] = image_size[1]
    return {"train": train_df, "validation": val_df, "test": test_df, "all": records}


def _safe_image_stem(row: pd.Series) -> str:
    return f"case_{int(row['case_id']):03d}_{str(row['image_id']).replace(' ', '_')}"


def export_processed_dataset(
    dataset_root: Optional[str | Path] = None,
    output_root: str | Path = "processed_annocerv",
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    split_config: SplitConfig = SplitConfig(),
    overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_root} already exists and is not empty. Use overwrite=True to replace files.")

    # Export raw pixel values; normalisation and augmentation belong to the training pipeline.
    output_root.mkdir(parents=True, exist_ok=True)
    splits = prepare_splits(dataset_root=dataset_root, image_size=image_size, split_config=split_config)
    processed_frames: Dict[str, pd.DataFrame] = {}

    for split_name in ["train", "validation", "test"]:
        split_dir = output_root / split_name
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        class_mask_dir = split_dir / "class_masks"
        for directory in [image_dir, mask_dir, class_mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        for _, row in splits[split_name].iterrows():
            stem = _safe_image_stem(row)
            image_out = image_dir / f"{stem}.png"
            mask_out = mask_dir / f"{stem}_mask.png"
            class_mask_out = class_mask_dir / f"{stem}_classes.png"

            image = read_rgb_image(row["image_path"], image_size=image_size)
            binary_mask = read_annotation_mask(row["mask_path"], image_size=image_size, binary=True)
            class_mask = read_annotation_mask(row["mask_path"], image_size=image_size, binary=False)

            Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), mode="RGB").save(image_out)
            Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L").save(mask_out)
            Image.fromarray(class_mask.astype(np.uint8), mode="L").save(class_mask_out)

            processed_row = row.to_dict()
            processed_row.update(
                {
                    "split": split_name,
                    "image_path": str(image_out.relative_to(output_root)),
                    "mask_path": str(mask_out.relative_to(output_root)),
                    "class_mask_path": str(class_mask_out.relative_to(output_root)),
                    "processed_root": str(output_root),
                    "mask_format": "binary_png",
                    "class_mask_format": "label_png",
                    "target_height": image_size[0],
                    "target_width": image_size[1],
                    "mask_area_processed": int(binary_mask.sum()),
                    "has_positive_mask": bool(binary_mask.sum() > 0),
                }
            )
            rows.append(processed_row)

        frame = pd.DataFrame(rows)
        frame.to_csv(split_dir / "metadata.csv", index=False)
        processed_frames[split_name] = frame

    all_frame = pd.concat([processed_frames["train"], processed_frames["validation"], processed_frames["test"]], ignore_index=True)
    all_frame.to_csv(output_root / "metadata.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "image_height": image_size[0],
                "image_width": image_size[1],
                "n_images": len(all_frame),
                "n_cases": all_frame["case_id"].nunique(),
                "train_images": len(processed_frames["train"]),
                "validation_images": len(processed_frames["validation"]),
                "test_images": len(processed_frames["test"]),
            }
        ]
    )
    manifest.to_csv(output_root / "manifest.csv", index=False)
    processed_frames["all"] = all_frame
    return processed_frames


def load_processed_splits(processed_root: Optional[str | Path] = None) -> Dict[str, pd.DataFrame]:

    root = resolve_processed_dataset_root(processed_root, required=True)
    metadata_path = root / "metadata.csv"
    if metadata_path.exists():
        all_frame = pd.read_csv(metadata_path)
    else:
        frames = []
        for split_name in ["train", "validation", "test"]:
            split_metadata = root / split_name / "metadata.csv"
            if not split_metadata.exists():
                raise FileNotFoundError(f"Missing metadata file: {split_metadata}")
            frames.append(pd.read_csv(split_metadata))
        all_frame = pd.concat(frames, ignore_index=True)

    def absolutize(path_value: object) -> str:
        path = Path(str(path_value))
        return str(path if path.is_absolute() else root / path)

    for column in ["image_path", "mask_path", "class_mask_path"]:
        if column in all_frame.columns:
            all_frame[column] = all_frame[column].map(absolutize)

    if "mask_format" not in all_frame.columns:
        all_frame["mask_format"] = "binary_png"
    if "split" not in all_frame.columns:
        raise ValueError("Processed metadata must contain a split column.")

    splits = {
        "train": all_frame[all_frame["split"] == "train"].copy().reset_index(drop=True),
        "validation": all_frame[all_frame["split"] == "validation"].copy().reset_index(drop=True),
        "test": all_frame[all_frame["split"] == "test"].copy().reset_index(drop=True),
        "all": all_frame.copy().reset_index(drop=True),
    }
    return splits


def prepare_data(
    dataset_root: Optional[str | Path] = None,
    processed_root: Optional[str | Path] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    split_config: SplitConfig = SplitConfig(),
    prefer_processed: bool = True,
) -> Dict[str, pd.DataFrame]:

    if prefer_processed:
        resolved_processed = resolve_processed_dataset_root(processed_root, required=False)
        if resolved_processed is not None:
            return load_processed_splits(resolved_processed)
    return prepare_splits(dataset_root=dataset_root, image_size=image_size, split_config=split_config)


class SegmentationAugmenter:

    def __init__(
        self,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.15,
        rotate90_p: float = 0.35,
        brightness_contrast_p: float = 0.35,
        noise_p: float = 0.15,
    ) -> None:
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotate90_p = rotate90_p
        self.brightness_contrast_p = brightness_contrast_p
        self.noise_p = noise_p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Apply every spatial transform to both arrays to keep annotations aligned.
        if random.random() < self.horizontal_flip_p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < self.vertical_flip_p:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < self.rotate90_p:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        if random.random() < self.brightness_contrast_p:
            brightness = random.uniform(-0.08, 0.08)
            contrast = random.uniform(0.85, 1.15)
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        if random.random() < self.noise_p:
            image = np.clip(image + np.random.normal(0.0, 0.015, image.shape).astype(np.float32), 0.0, 1.0)
        return image.astype(np.float32), mask.astype(np.float32)


class AnnoCervSegmentationDataset(Dataset):

    def __init__(
        self,
        records: pd.DataFrame,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        return_metadata: bool = True,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.image_size = image_size
        self.augmenter = SegmentationAugmenter() if augment else None
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AnnoCervSegmentationDataset. Install torch or run this part on Kaggle GPU.")
        row = self.records.iloc[idx]
        image = read_rgb_image(row["image_path"], self.image_size)
        mask_format = str(row.get("mask_format", "annotation_png"))
        if mask_format == "binary_png":
            mask = read_binary_mask_image(row["mask_path"], self.image_size)
        else:
            mask = read_annotation_mask(row["mask_path"], self.image_size, binary=True)

        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        # Convert from image layout (H, W, C) to the channel-first layout expected by PyTorch.
        image = normalize_image(image, self.mean, self.std)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()

        sample: Dict[str, object] = {"image": image_tensor, "mask": mask_tensor}
        if self.return_metadata:
            sample["metadata"] = {
                "image_id": row["image_id"],
                "case_id": int(row["case_id"]),
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "swede_score": float("nan") if pd.isna(row["swede_score"]) else float(row["swede_score"]),
                "original_width": int(row["original_width"]),
                "original_height": int(row["original_height"]),
            }
        return sample


def create_datasets(
    splits: Dict[str, pd.DataFrame],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Dict[str, AnnoCervSegmentationDataset]:

    return {
        "train": AnnoCervSegmentationDataset(splits["train"], image_size=image_size, augment=True),
        "validation": AnnoCervSegmentationDataset(splits["validation"], image_size=image_size, augment=False),
        "test": AnnoCervSegmentationDataset(splits["test"], image_size=image_size, augment=False),
    }


def create_dataloaders(
    splits: Dict[str, pd.DataFrame],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to create DataLoaders. The processed dataset export does not require torch.")
    datasets = create_datasets(splits, image_size=image_size)
    if pin_memory is None:
        # Pinned host memory is useful for GPU transfers but unnecessary on CPU-only runs.
        pin_memory = torch.cuda.is_available()
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> np.ndarray:

    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    mean_arr = np.asarray(mean).reshape(1, 1, 3)
    std_arr = np.asarray(std).reshape(1, 1, 3)
    return np.clip(array * std_arr + mean_arr, 0.0, 1.0)


def resize_mask_tensor(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to resize tensors.")
    return F.interpolate(mask.float(), size=size, mode="nearest")


def dataset_summary(records: pd.DataFrame) -> Dict[str, object]:

    positive = records["has_positive_mask"].sum()
    swede_counts = records["swede_score"].value_counts(dropna=False).sort_index()
    return {
        "n_images": int(len(records)),
        "n_cases": int(records["case_id"].nunique()),
        "n_masks": int(records["has_mask"].sum()),
        "n_positive_masks": int(positive),
        "n_empty_masks": int(len(records) - positive),
        "image_width_summary": records["original_width"].describe().to_dict(),
        "image_height_summary": records["original_height"].describe().to_dict(),
        "swede_distribution": swede_counts.to_dict(),
    }


def class_pixel_distribution(
    records: pd.DataFrame,
    image_size: Optional[Tuple[int, int]] = None,
    max_items: Optional[int] = None,
) -> pd.DataFrame:

    rows: List[Dict[str, object]] = []
    iterator = records.head(max_items) if max_items is not None else records
    for _, row in iterator.iterrows():
        if "class_mask_path" in row and isinstance(row.get("class_mask_path"), str) and Path(row["class_mask_path"]).exists():
            labels = read_class_mask_image(row["class_mask_path"], image_size=image_size)
        else:
            labels = read_annotation_mask(row["mask_path"], image_size=image_size, binary=False)
        counts = {INDEX_TO_CLASS[idx]: int((labels == idx).sum()) for idx in INDEX_TO_CLASS}
        counts.update({"image_id": row["image_id"], "case_id": row["case_id"]})
        rows.append(counts)
    return pd.DataFrame(rows)


def sample_records(records: pd.DataFrame, n: int = 6, random_state: int = 42) -> pd.DataFrame:

    positive = records[records["has_positive_mask"]]
    source = positive if len(positive) >= n else records
    return source.sample(n=min(n, len(source)), random_state=random_state).reset_index(drop=True)


__all__ = [
    "ANNOTATION_COLORS",
    "CLASS_TO_INDEX",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "INDEX_TO_CLASS",
    "AnnoCervSegmentationDataset",
    "SegmentationAugmenter",
    "SplitConfig",
    "class_pixel_distribution",
    "collect_annocerv_records",
    "create_dataloaders",
    "create_datasets",
    "dataset_summary",
    "denormalize_tensor",
    "export_processed_dataset",
    "load_processed_splits",
    "load_swede_scores",
    "normalize_image",
    "parse_annocerv_filename",
    "prepare_data",
    "prepare_splits",
    "read_annotation_mask",
    "read_binary_mask_image",
    "read_class_mask_image",
    "read_rgb_image",
    "resize_mask_tensor",
    "resolve_dataset_root",
    "resolve_processed_dataset_root",
    "sample_records",
    "seed_everything",
    "split_records",
]
