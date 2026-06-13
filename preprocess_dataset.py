#!/usr/bin/env python3

from pathlib import Path
import argparse
import random
import shutil
from PIL import Image
import sys
import csv


def is_image_file(p: Path):
    return p.is_file() and p.suffix.lower() in (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    )


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, left + m))


def preprocess_image(img: Image.Image, image_size: int, crop_mode: str):
    img = img.convert("RGB")

    if crop_mode == "center":
        img = center_crop_square(img)

    img = img.resize((image_size, image_size), Image.LANCZOS)
    return img


def save_image(src: Path, dst: Path, image_size: int, crop_mode: str):
    try:
        with Image.open(src) as img:
            img = preprocess_image(img, image_size, crop_mode)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, quality=95)
    except Exception as e:
        print(f"[ERROR] {src} -> {e}")


def get_label_from_path(path: Path):
    p = str(path).lower()

    if "high_grade" in p:
        return 1
    if "low_grade" in p:
        return 0

    return None


def collect_patients(root: Path):
    patients = {}

    case_dirs = list(root.rglob("case_*"))

    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue

        images = [p for p in case_dir.rglob("*") if is_image_file(p)]

        if images:
            patients[case_dir.name] = sorted(images)

    if len(patients) == 0:
        print("[ERROR] No case_* folders found.")
        sys.exit(2)

    return patients


def patient_split(keys, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = keys[:n_train]
    val = keys[n_train:n_train + n_val]
    test = keys[n_train + n_val:]

    return train, val, test


def copy_patient(images, out_dir: Path, image_size, crop_mode):
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in images:
        dst = out_dir / src.name
        save_image(src, dst, image_size, crop_mode)


def write_labels(patients, split_ids, split_name, dest_root: Path):
    csv_path = dest_root / f"{split_name}_labels.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "label"])

        for pid in split_ids:
            first_img = patients[pid][0]
            label = get_label_from_path(first_img)
            writer.writerow([pid, label])

    print(f"[INFO] labels saved -> {csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-mode", choices=["center", "none"], default="center")

    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    src_root = Path(args.source)
    dest_root = Path(args.dest)

    if not src_root.exists():
        print("Source not found:", src_root)
        sys.exit(1)

    if dest_root.exists():
        shutil.rmtree(dest_root)

    dest_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Collecting patients...")
    patients = collect_patients(src_root)

    print(f"[INFO] Total patients: {len(patients)}")

    train_ids, val_ids, test_ids = patient_split(
        patients.keys(),
        args.train,
        args.val,
        args.seed
    )

    print(f"[INFO] Split -> train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}")

    for split in ["train", "val", "test"]:
        (dest_root / split).mkdir(parents=True, exist_ok=True)

    for pid in train_ids:
        copy_patient(patients[pid], dest_root / "train" / pid, args.image_size, args.crop_mode)

    for pid in val_ids:
        copy_patient(patients[pid], dest_root / "val" / pid, args.image_size, args.crop_mode)

    for pid in test_ids:
        copy_patient(patients[pid], dest_root / "test" / pid, args.image_size, args.crop_mode)

    write_labels(patients, train_ids, "train", dest_root)
    write_labels(patients, val_ids, "val", dest_root)
    write_labels(patients, test_ids, "test", dest_root)

    print("[DONE]", dest_root)


if __name__ == "__main__":
    main()