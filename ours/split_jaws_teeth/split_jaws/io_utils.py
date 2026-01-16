# split_jaws/io_utils.py
import os
import numpy as np

def load_mask_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr

def save_mask_npy(path: str, mask_uint8: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, mask_uint8.astype(np.uint8, copy=False))

def ensure_binary_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)
