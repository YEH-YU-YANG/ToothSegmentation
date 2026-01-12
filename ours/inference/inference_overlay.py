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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="實驗名稱")
    parser.add_argument("--input", type=str, required=True, help="輸入影像資料夾")
    parser.add_argument("--output", type=str, required=True, help="輸出資料夾")
    parser.add_argument("--fold", type=int, default=1, help="Fold")
    parser.add_argument("--alpha", type=float, default=0.3, help="透明度")
    parser.add_argument("--patient", type=str, default="", help="病患名稱")
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

    print(f"Processing Overlay: Found {len(image_paths)} images")
    print(f"Flip: lr={args.flip_lr}, ud={args.flip_ud}")

    for image_path in track(image_paths, desc=args.patient if args.patient else "Overlay"):
        image_tensor = CBCTDataset.load_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(config.device)

        with torch.no_grad(), torch.autocast(config.device):
            predicts = model(image_tensor)
            if isinstance(predicts, tuple):
                predicts = predicts[0]

        pred_mask = predicts.argmax(1).cpu().numpy()[0].astype(np.uint8)

        original_cv = cv2.imread(image_path)
        if original_cv is None:
            continue

        if args.flip_lr or args.flip_ud:
            pred_mask = _apply_flips_2d(pred_mask, flip_lr=args.flip_lr, flip_ud=args.flip_ud)
            if args.flip_lr:
                original_cv = original_cv[:, ::-1, :]
            if args.flip_ud:
                original_cv = original_cv[::-1, :, :]

        color_mask = np.zeros_like(original_cv)
        color_mask[pred_mask == 1] = [0, 0, 255]

        overlay = cv2.addWeighted(original_cv, 1, color_mask, args.alpha, 0)

        rel_path = os.path.relpath(image_path, args.input)
        output_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
