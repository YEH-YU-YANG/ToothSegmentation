import os
import glob
import re
import torch
import cv2
import numpy as np
from argparse import ArgumentParser

from src.config import load_config
from src.models import load_model
from src.console import track
from src.dataset import CBCTDataset
from src.downloader import ensure_experiment_exists


def _natural_key(s: str):
    # Natural sort: "2.png" < "10.png"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _collect_images(input_dir: str):
    extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    # 去重 + 以「相對路徑」做自然排序，確保堆疊順序穩定
    image_paths = list(set(image_paths))
    image_paths.sort(key=lambda p: _natural_key(os.path.relpath(p, input_dir)))
    return image_paths


def _apply_flips_2d(arr_hw: np.ndarray, flip_lr: bool, flip_ud: bool):
    if flip_lr:
        arr_hw = arr_hw[:, ::-1]
    if flip_ud:
        arr_hw = arr_hw[::-1, :]
    return arr_hw


def _apply_flips_3d(vol_dhw: np.ndarray, flip_z: bool, flip_lr: bool, flip_ud: bool):
    # vol shape: (D, H, W)
    if flip_z:
        vol_dhw = np.flip(vol_dhw, axis=0)
    if flip_ud:
        vol_dhw = np.flip(vol_dhw, axis=1)
    if flip_lr:
        vol_dhw = np.flip(vol_dhw, axis=2)
    return vol_dhw


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="實驗名稱")
    parser.add_argument("--input", type=str, required=True, help="輸入影像資料夾 (PNG)")
    parser.add_argument("--output", type=str, required=True, help="PNG Mask 輸出資料夾")
    parser.add_argument("--output_npy", type=str, default=None, help="NPY 檔案輸出資料夾 (預設同 output)")
    parser.add_argument("--fold", type=int, default=1, help="Fold")
    parser.add_argument("--patient", type=str, default="", help="病患名稱 (用於檔名/顯示)")

    # === 視角修正（預設全關：你的 input PNG 已經是 STL 視角）===
    parser.add_argument("--flip_z", action="store_true", help="翻轉切片順序 (Depth/Z)")
    parser.add_argument("--flip_lr", action="store_true", help="左右翻轉 (Width/X)")
    parser.add_argument("--flip_ud", action="store_true", help="上下翻轉 (Height/Y)")

    args = parser.parse_args()

    ensure_experiment_exists(args.exp)
    config = load_config(os.path.join("logs", args.exp, "config.toml"))
    config.fold = args.fold
    model = load_model(config)

    image_paths = _collect_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    npy_dest = args.output_npy if args.output_npy else args.output

    print(f"Found {len(image_paths)} images.")
    print(f"PNG Output (Class 1 Only) -> {args.output}")
    print(f"NPY Output (Both Binary & Multi) -> {npy_dest}")
    print(f"Flip: z={args.flip_z}, lr={args.flip_lr}, ud={args.flip_ud}")

    mask_buffer = []

    for image_path in track(image_paths, desc=args.patient if args.patient else "Inference"):
        image_tensor = CBCTDataset.load_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(config.device)

        with torch.no_grad(), torch.autocast(config.device):
            predicts = model(image_tensor)
            if isinstance(predicts, tuple):
                predicts = predicts[0]

        pred_mask = predicts.argmax(1).cpu().numpy()[0].astype(np.uint8)

        # 對單張 slice 做 LR/UD 翻轉（若有需要）
        pred_mask = _apply_flips_2d(pred_mask, flip_lr=args.flip_lr, flip_ud=args.flip_ud)

        mask_buffer.append(pred_mask)

        # 存 PNG（只顯示 class 1）
        png_viz = np.zeros_like(pred_mask, dtype=np.uint8)
        png_viz[pred_mask == 1] = 255

        rel_path = os.path.relpath(image_path, args.input)
        output_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, png_viz)

    if not mask_buffer:
        raise RuntimeError("mask_buffer is empty (no slices processed).")

    volume = np.stack(mask_buffer, axis=0)  # (D,H,W)

    # 只在這裡做 flip_z（避免逐張影響速度）
    volume = _apply_flips_3d(volume, flip_z=args.flip_z, flip_lr=False, flip_ud=False)

    os.makedirs(npy_dest, exist_ok=True)

    multi_filename = f"{args.patient}_mask_multi.npy" if args.patient else "mask_multi.npy"
    multi_path = os.path.join(npy_dest, multi_filename)
    np.save(multi_path, volume)
    print(f"[1/2] Saved Multi-Class -> {multi_path}  shape={volume.shape}")

    binary_volume = (volume == 1).astype(np.uint8)
    binary_filename = f"{args.patient}_mask_binary.npy" if args.patient else "mask_binary.npy"
    binary_path = os.path.join(npy_dest, binary_filename)
    np.save(binary_path, binary_volume)
    print(f"[2/2] Saved Binary      -> {binary_path}  shape={binary_volume.shape}")
