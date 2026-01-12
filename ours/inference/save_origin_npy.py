import os
import glob
import re
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _collect_images(input_dir: str):
    extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

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
    if flip_z:
        vol_dhw = np.flip(vol_dhw, axis=0)
    if flip_ud:
        vol_dhw = np.flip(vol_dhw, axis=1)
    if flip_lr:
        vol_dhw = np.flip(vol_dhw, axis=2)
    return vol_dhw


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="輸入影像資料夾 (PNG)")
    parser.add_argument("--output", type=str, required=True, help="輸出資料夾")
    parser.add_argument("--patient", type=str, default="", help="病患名稱 (用於檔名)")
    # 預設不翻：你的 input PNG 已經是 STL 視角
    parser.add_argument("--flip_z", action="store_true", help="翻轉切片順序 (Depth/Z)")
    parser.add_argument("--flip_lr", action="store_true", help="左右翻轉 (Width/X)")
    parser.add_argument("--flip_ud", action="store_true", help="上下翻轉 (Height/Y)")
    args = parser.parse_args()

    image_paths = _collect_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    print(f"Found {len(image_paths)} images. Building origin volume...")
    print(f"Flip: z={args.flip_z}, lr={args.flip_lr}, ud={args.flip_ud}")

    buf = []
    for path in tqdm(image_paths, desc=f"Loading {args.patient}" if args.patient else "Loading"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {path}")
            continue
        img = _apply_flips_2d(img, flip_lr=args.flip_lr, flip_ud=args.flip_ud)
        buf.append(img)

    if not buf:
        raise RuntimeError("Buffer is empty.")

    vol = np.stack(buf, axis=0)  # (D,H,W)
    vol = _apply_flips_3d(vol, flip_z=args.flip_z, flip_lr=False, flip_ud=False)

    os.makedirs(args.output, exist_ok=True)
    filename = f"{args.patient}_origin.npy" if args.patient else "origin.npy"
    save_path = os.path.join(args.output, filename)
    np.save(save_path, vol)

    print(f"Saved: {save_path}")
    print(f"Shape: {vol.shape}, dtype: {vol.dtype}")


if __name__ == "__main__":
    main()
