#!/usr/bin/env python
"""
清洗指定病人的 mask_binary.npy 和 mask_multi.npy
輸出檔名會加上 _clean 後綴
"""

import numpy as np
import os
import sys
from scipy import ndimage


def remove_small_components_3d(bin_mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """
    移除小於指定體素數的連通區域

    Parameters:
    -----------
    bin_mask : np.ndarray
        3D 二值遮罩
    min_voxels : int
        最小體素數閾值，小於此值的連通區域將被移除

    Returns:
    --------
    np.ndarray
        清洗後的二值遮罩
    """
    if min_voxels is None or min_voxels <= 0:
        return bin_mask.astype(bool)

    structure = ndimage.generate_binary_structure(3, 2)  # 26-connected
    labeled, num = ndimage.label(bin_mask, structure=structure)

    if num == 0:
        return bin_mask.astype(bool)

    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[1:] = counts[1:] >= int(min_voxels)

    return keep[labeled]


def keep_top_k_components_3d(bin_mask: np.ndarray, k: int) -> np.ndarray:
    """
    只保留最大的 k 個連通區域

    Parameters:
    -----------
    bin_mask : np.ndarray
        3D 二值遮罩
    k : int
        要保留的最大連通區域數量

    Returns:
    --------
    np.ndarray
        只包含最大 k 個連通區域的二值遮罩
    """
    if k is None or k <= 0:
        return bin_mask.astype(bool)

    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(bin_mask, structure=structure)

    if num == 0:
        return bin_mask.astype(bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # 忽略背景
    top = np.argsort(counts)[::-1][: int(k)]

    return np.isin(labeled, top)

def get_component_info(bin_mask: np.ndarray) -> dict:
    """
    獲取連通區域的統計資訊

    Parameters:
    -----------
    bin_mask : np.ndarray
        3D 二值遮罩

    Returns:
    --------
    dict
        包含連通區域資訊的字典
    """
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(bin_mask, structure=structure)

    if num == 0:
        return {
            'num_components': 0,
            'total_voxels': 0,
            'component_sizes': [],
            'largest_size': 0,
            'smallest_size': 0
        }

    counts = np.bincount(labeled.ravel())
    sizes = counts[1:]  # 排除背景

    return {
        'num_components': num,
        'total_voxels': int(bin_mask.sum()),
        'component_sizes': sorted(sizes.tolist(), reverse=True),
        'largest_size': int(sizes.max()),
        'smallest_size': int(sizes.min()),
        'mean_size': float(sizes.mean()),
        'median_size': float(np.median(sizes))
    }

def clean_mask(
    mask: np.ndarray,
    min_global: int = None,
    keep_top_k_global: int = None,
    verbose: bool = True
) -> np.ndarray:
    """
    完整的全域清洗流程

    Parameters:
    -----------
    mask : np.ndarray
        3D 遮罩（可以是二值或整數標籤）
    min_global : int, optional
        全域最小體素數閾值
    keep_top_k_global : int, optional
        全域保留的最大連通區域數量
    verbose : bool
        是否顯示處理訊息

    Returns:
    --------
    np.ndarray
        清洗後的二值遮罩
    """
    teeth = (mask > 0).astype(bool)

    if verbose:
        initial_voxels = int(teeth.sum())
        print(f"Initial voxels: {initial_voxels:,}")

    # Step 1: 移除小連通區域
    if min_global is not None and min_global > 0:
        teeth = remove_small_components_3d(teeth, min_global)
        if verbose:
            after_min = int(teeth.sum())
            print(f"After removing small components (min={min_global}): {after_min:,} voxels")

    # Step 2: 只保留最大的 k 個連通區域
    if keep_top_k_global is not None and keep_top_k_global > 0:
        teeth = keep_top_k_components_3d(teeth, keep_top_k_global)
        if verbose:
            after_topk = int(teeth.sum())
            print(f"After keeping top {keep_top_k_global} components: {after_topk:,} voxels")

    return teeth

def clean_patient_masks(patient_dir, min_voxels=5000, keep_top_k=2):
    """
    清洗病人資料夾中的遮罩檔案

    Parameters:
    -----------
    patient_dir : str
        病人資料夾路徑 (例如: "57969132")
    min_voxels : int
        最小體素數閾值
    keep_top_k : int
        保留最大的 K 個連通區域
    """

    # 檢查資料夾是否存在
    from pathlib import Path

    # 定義根目錄
    base_dir = Path("/home/p76144736/yeh/paper_related_project/ToothSegmentation/data")

    # 拼接路徑 (使用 / 運算子)
    patient_path = base_dir / patient_dir
    if not os.path.isdir(patient_path):
        print(f"錯誤: 找不到資料夾 {patient_path}")
        return False

    # 要處理的檔案
    files_to_clean = [
        ("mask_binary.npy", "mask_binary_clean.npy"),
        ("mask_multi.npy", "mask_multi_clean.npy")
    ]

    for input_file, output_file in files_to_clean:
        input_path = os.path.join(patient_path, input_file)
        output_path = os.path.join(patient_path, output_file)

        # 檢查檔案是否存在
        if not os.path.exists(input_path):
            print(f"\n警告: 找不到檔案 {input_path}，跳過")
            continue

        print(f"\n{'='*60}")
        print(f"處理檔案: {input_file}")
        print(f"{'='*60}")

        # 載入遮罩
        try:
            mask = np.load(input_path)
            print(f"遮罩形狀: {mask.shape}, dtype: {mask.dtype}")
        except Exception as e:
            print(f"錯誤: 無法載入 {input_path}")
            print(f"錯誤訊息: {e}")
            continue

        # 顯示原始資訊
        print(f"\n--- 原始遮罩統計 ---")
        info_before = get_component_info(mask > 0)
        print(f"連通區域數量: {info_before['num_components']}")
        print(f"總體素數: {info_before['total_voxels']:,}")

        if info_before['num_components'] > 0:
            print(f"最大區域: {info_before['largest_size']:,} 體素")
            print(f"最小區域: {info_before['smallest_size']:,} 體素")
            if info_before['num_components'] <= 10:
                print(f"所有區域大小: {info_before['component_sizes']}")
            else:
                print(f"前 10 大區域: {info_before['component_sizes'][:10]}")

        # 清洗遮罩
        print(f"\n--- 開始清洗 ---")
        print(f"參數: min_voxels={min_voxels}, keep_top_k={keep_top_k}")

        cleaned = clean_mask(
            mask,
            min_global=min_voxels,
            keep_top_k_global=keep_top_k,
            verbose=True
        )

        # 顯示清洗後資訊
        print(f"\n--- 清洗後統計 ---")
        info_after = get_component_info(cleaned)
        print(f"連通區域數量: {info_after['num_components']}")
        print(f"總體素數: {info_after['total_voxels']:,}")

        if info_after['num_components'] > 0:
            print(f"最大區域: {info_after['largest_size']:,} 體素")
            print(f"最小區域: {info_after['smallest_size']:,} 體素")
            print(f"區域大小: {info_after['component_sizes']}")

        # 計算變化
        voxels_removed = info_before['total_voxels'] - info_after['total_voxels']
        percent_kept = (info_after['total_voxels'] / info_before['total_voxels'] * 100) if info_before['total_voxels'] > 0 else 0

        print(f"\n移除體素數: {voxels_removed:,} ({100-percent_kept:.2f}%)")
        print(f"保留體素數: {info_after['total_voxels']:,} ({percent_kept:.2f}%)")

        # 儲存結果
        try:
            np.save(output_path, cleaned.astype(np.uint8))
            print(f"\n✓ 已儲存至: {output_path}")
        except Exception as e:
            print(f"\n✗ 儲存失敗: {e}")
            continue

    print(f"\n{'='*60}")
    print("處理完成！")
    print(f"{'='*60}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="清洗病人遮罩檔案")
    parser.add_argument("patient", type=str, help="病人資料夾路徑 (例如: 57969132)")
    parser.add_argument("--min_voxels", type=int, default=5000, 
                       help="最小體素數閾值 (預設: 5000)")
    parser.add_argument("--keep_top_k", type=int, default=0, 
                       help="保留最大的 K 個連通區域 (預設: 2，適合上下顎)")

    args = parser.parse_args()

    # 執行清洗
    success = clean_patient_masks(
        args.patient,
        min_voxels=args.min_voxels,
        keep_top_k=args.keep_top_k
    )

    sys.exit(0 if success else 1)
