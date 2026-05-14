# -*- coding: utf-8 -*-
"""
Create binary datasets for SmartAppAttack reproduction.

Folders are compatible with the original repo:
../datasets/<DATASET>/train_stop_sim/non_stop, stop
../datasets/<DATASET>/test_stop_sim/non_stop, stop
../datasets/<DATASET>/train_ebama_sim/non_stop, stop
../datasets/<DATASET>/test_ebama_sim/non_stop, stop
../datasets/<DATASET>/train_sim/non_stop, stop
../datasets/<DATASET>/valid_sim/non_stop, stop
../datasets/<DATASET>/test_sim/non_stop, stop

Class name convention:
- stop     = target class, numeric label 1 in Keras folder loading
- non_stop = non-target class, numeric label 0
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_CONFIGS = {
    "GTSRB": {
        "target_label": 14,             # Stop
        "bama_non_target_label": 0,
        "ebama_non_target_label": 9,    # No passing
        "target_name": "Stop",
        "error_prone_name": "No passing",
        "tfds_name": "gtsrb",
    },
    "CIFAR10": {
        "target_label": 0,              # Airplane
        "bama_non_target_label": 1,     # Automobile
        "ebama_non_target_label": 9,    # Truck
        "target_name": "Airplane",
        "error_prone_name": "Truck",
        "tfds_name": None,
    },
    "FLOWERS": {
        # TFDS Oxford Flowers labels are integer IDs; this package uses a stable fallback.
        "target_label": 0,
        "bama_non_target_label": 1,
        "ebama_non_target_label": 1,
        "target_name": "OxfordFlowers target",
        "error_prone_name": "OxfordFlowers non-target",
        "tfds_name": "oxford_flowers102",
    },
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}

def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def ensure_class_dirs(root):
    (root / "stop").mkdir(parents=True, exist_ok=True)
    (root / "non_stop").mkdir(parents=True, exist_ok=True)

def save_array(arr, out_path):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")

def save_file(src, out_path):
    img = Image.open(src)
    if img.mode != "RGB":
        img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")

def sample_list(items, n):
    items = list(items)
    random.shuffle(items)
    return items[:min(n, len(items))]

def write_arrays(split_root, x, y, target_label, non_label, nt, nn):
    split_root = Path(split_root)
    reset_dir(split_root)
    ensure_class_dirs(split_root)
    y = np.asarray(y).reshape(-1)
    target_idx = sample_list(np.where(y == target_label)[0], nt)
    non_idx = sample_list(np.where(y == non_label)[0], nn)
    for k, idx in enumerate(target_idx):
        save_array(x[idx], split_root / "stop" / "stop_{:05d}.png".format(k))
    for k, idx in enumerate(non_idx):
        save_array(x[idx], split_root / "non_stop" / "non_stop_{:05d}.png".format(k))
    return len(target_idx), len(non_idx)

def build_cifar10(out_root, cfg, train_target, train_non, test_target, test_non):
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    stats = {}
    stats["train_stop_sim"] = write_arrays(out_root / "train_stop_sim", x_train, y_train, cfg["target_label"], cfg["bama_non_target_label"], train_target, train_non)
    stats["test_stop_sim"] = write_arrays(out_root / "test_stop_sim", x_test, y_test, cfg["target_label"], cfg["bama_non_target_label"], test_target, test_non)
    stats["train_ebama_sim"] = write_arrays(out_root / "train_ebama_sim", x_train, y_train, cfg["target_label"], cfg["ebama_non_target_label"], train_target, train_non)
    stats["test_ebama_sim"] = write_arrays(out_root / "test_ebama_sim", x_test, y_test, cfg["target_label"], cfg["ebama_non_target_label"], test_target, test_non)
    stats["train_sim"] = write_arrays(out_root / "train_sim", x_train, y_train, cfg["target_label"], cfg["bama_non_target_label"], int(train_target * 0.8), int(train_non * 0.8))
    stats["valid_sim"] = write_arrays(out_root / "valid_sim", x_train, y_train, cfg["target_label"], cfg["bama_non_target_label"], max(20, int(train_target * 0.2)), max(20, int(train_non * 0.2)))
    stats["test_sim"] = write_arrays(out_root / "test_sim", x_test, y_test, cfg["target_label"], cfg["bama_non_target_label"], test_target, test_non)
    return stats

def build_gtsrb_local(out_root, cfg, train_target, train_non, test_target, test_non):
    raw_candidates = [
        Path("../datasets/raw_gtsrb"),
        Path("../datasets/GTSRB/raw"),
        Path("../datasets/raw/GTSRB"),
    ]
    class_to_files = {}
    for raw in raw_candidates:
        if not raw.exists():
            continue
        for f in raw.rglob("*"):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            try:
                lab = int(f.parent.name)
            except Exception:
                continue
            class_to_files.setdefault(lab, []).append(f)
    if cfg["target_label"] not in class_to_files:
        raise FileNotFoundError("Local GTSRB raw data not found; falling back to TFDS.")

    def write_files(split_root, target_label, non_label, nt, nn):
        split_root = Path(split_root)
        reset_dir(split_root)
        ensure_class_dirs(split_root)
        t_files = sample_list(class_to_files.get(target_label, []), nt)
        n_files = sample_list(class_to_files.get(non_label, []), nn)
        for k, src in enumerate(t_files):
            save_file(src, split_root / "stop" / "stop_{:05d}.png".format(k))
        for k, src in enumerate(n_files):
            save_file(src, split_root / "non_stop" / "non_stop_{:05d}.png".format(k))
        return len(t_files), len(n_files)

    stats = {}
    stats["train_stop_sim"] = write_files(out_root / "train_stop_sim", cfg["target_label"], cfg["bama_non_target_label"], train_target, train_non)
    stats["test_stop_sim"] = write_files(out_root / "test_stop_sim", cfg["target_label"], cfg["bama_non_target_label"], test_target, test_non)
    stats["train_ebama_sim"] = write_files(out_root / "train_ebama_sim", cfg["target_label"], cfg["ebama_non_target_label"], train_target, train_non)
    stats["test_ebama_sim"] = write_files(out_root / "test_ebama_sim", cfg["target_label"], cfg["ebama_non_target_label"], test_target, test_non)
    stats["train_sim"] = write_files(out_root / "train_sim", cfg["target_label"], cfg["bama_non_target_label"], int(train_target * 0.8), int(train_non * 0.8))
    stats["valid_sim"] = write_files(out_root / "valid_sim", cfg["target_label"], cfg["bama_non_target_label"], max(20, int(train_target * 0.2)), max(20, int(train_non * 0.2)))
    stats["test_sim"] = write_files(out_root / "test_sim", cfg["target_label"], cfg["bama_non_target_label"], test_target, test_non)
    return stats

def build_from_tfds(out_root, cfg, train_target, train_non, test_target, test_non):
    import tensorflow_datasets as tfds
    train_ds, test_ds = tfds.load(cfg["tfds_name"], split=["train", "test"], as_supervised=True)

    def collect(ds, target_label, non_label, nt, nn):
        target, non = [], []
        for image, label in tfds.as_numpy(ds):
            lab = int(label)
            if lab == target_label and len(target) < nt:
                target.append(image)
            elif lab == non_label and len(non) < nn:
                non.append(image)
            if len(target) >= nt and len(non) >= nn:
                break
        return target, non

    def write_split(split_root, target_label, non_label, nt, nn):
        split_root = Path(split_root)
        reset_dir(split_root)
        ensure_class_dirs(split_root)
        target, non = collect(train_ds if "train" in split_root.name or "valid" in split_root.name else test_ds, target_label, non_label, nt, nn)
        for k, arr in enumerate(target):
            save_array(arr, split_root / "stop" / "stop_{:05d}.png".format(k))
        for k, arr in enumerate(non):
            save_array(arr, split_root / "non_stop" / "non_stop_{:05d}.png".format(k))
        return len(target), len(non)

    stats = {}
    stats["train_stop_sim"] = write_split(out_root / "train_stop_sim", cfg["target_label"], cfg["bama_non_target_label"], train_target, train_non)
    stats["test_stop_sim"] = write_split(out_root / "test_stop_sim", cfg["target_label"], cfg["bama_non_target_label"], test_target, test_non)
    stats["train_ebama_sim"] = write_split(out_root / "train_ebama_sim", cfg["target_label"], cfg["ebama_non_target_label"], train_target, train_non)
    stats["test_ebama_sim"] = write_split(out_root / "test_ebama_sim", cfg["target_label"], cfg["ebama_non_target_label"], test_target, test_non)
    stats["train_sim"] = write_split(out_root / "train_sim", cfg["target_label"], cfg["bama_non_target_label"], int(train_target * 0.8), int(train_non * 0.8))
    stats["valid_sim"] = write_split(out_root / "valid_sim", cfg["target_label"], cfg["bama_non_target_label"], max(20, int(train_target * 0.2)), max(20, int(train_non * 0.2)))
    stats["test_sim"] = write_split(out_root / "test_sim", cfg["target_label"], cfg["bama_non_target_label"], test_target, test_non)
    return stats

def build_dataset(dataset, quick=False):
    cfg = DATASET_CONFIGS[dataset]
    out_root = Path("../datasets") / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    train_target = 80 if quick else 800
    train_non = 80 if quick else 800
    test_target = 20 if quick else 200
    test_non = 20 if quick else 200

    if dataset == "CIFAR10":
        stats = build_cifar10(out_root, cfg, train_target, train_non, test_target, test_non)
    elif dataset == "GTSRB":
        try:
            stats = build_gtsrb_local(out_root, cfg, train_target, train_non, test_target, test_non)
        except Exception as e:
            print("Local GTSRB fallback reason:", e)
            stats = build_from_tfds(out_root, cfg, train_target, train_non, test_target, test_non)
    elif dataset == "FLOWERS":
        stats = build_from_tfds(out_root, cfg, train_target, train_non, test_target, test_non)
    else:
        raise ValueError(dataset)

    print("\n=== Building {} ===".format(dataset))
    print("Target class = {} ({})".format(cfg["target_label"], cfg["target_name"]))
    print("E-BAMA non-target = {} ({})".format(cfg["ebama_non_target_label"], cfg["error_prone_name"]))
    print("Saved to:", out_root)
    for k, (t, n) in stats.items():
        print("{:15s}: target={}, non_target={}".format(k, t, n))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GTSRB", choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    build_dataset(args.dataset, quick=args.quick)

if __name__ == "__main__":
    main()
