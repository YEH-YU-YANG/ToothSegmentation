"""split_jaws/blob_split.py

本模組負責把「疑似上下顎黏在一起」的單一 blob（ROI 內二值 mask）切成兩類：
  - 1 = lower
  - 2 = upper

主方法：seed-based thickness watershed
  - 先用 EDT（distance transform）找「厚核心」當種子
  - 種子分別要求落在 thr_z 兩側且離分界至少 seed_margin
  - 用 cost(EDT) + watershed_ift 把整坨分成上下兩類

本次整合：h-maxima watershed fallback
  - 當主方法因 seeds 不足而 FAIL 時，嘗試用 h-maxima 在 EDT 上找多個峰
  - 先把 blob 切成多個子區塊，再依每個子區塊的 z-centroid（全域座標）
    併回 2 類（lower/upper）
  - 若 fallback 也無法得到兩邊都非空的分割，回傳全 0 讓 caller 做 centroid fallback
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .cc_utils import get_structure
from .watershed_cost import make_cost_from_edt


def _relax_seed_dt(edt_vals: np.ndarray, seed_dt_min: float, percentile: float) -> float:
    if edt_vals.size == 0:
        return seed_dt_min
    p = float(np.percentile(edt_vals, percentile))
    return max(seed_dt_min, p)


def _split_blob_hmax_fallback(
    blob: np.ndarray,
    *,
    z_global_offset: int,
    thr_z: float,
    connectivity: int,
    # h-maxima params
    hmax_h: float = 3.0,
    hmax_sigma: float = 1.0,
    hmax_pad: int = 10,
    hmax_min_markers: int = 2,
    # acceptance checks
    min_side_vox: int = 80,
    debug: bool = False,
    debug_prefix: str = "",
) -> np.ndarray:
    """h-maxima watershed fallback.

    回傳 ROI coords label map：1=lower,2=upper,0=背景。
    若失敗（markers 不夠/切不出兩邊），回傳全 0。
    """

    # lazy import：避免沒有 skimage 的環境直接炸掉主流程
    try:
        from skimage.morphology import h_maxima
        from skimage.segmentation import watershed
    except Exception:
        if debug:
            print(f"[blob_split]{debug_prefix}hmax fallback unavailable (no skimage)", flush=True)
        return np.zeros_like(blob, dtype=np.uint8)

    blob_b = (blob > 0)
    if not blob_b.any():
        return np.zeros_like(blob, dtype=np.uint8)

    # Tight bbox + pad（在 caller ROI 內再切一次小 ROI，加速）
    coords = np.argwhere(blob_b)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    Z, Y, X = blob_b.shape
    pad = int(hmax_pad)
    z0p = max(z0 - pad, 0)
    y0p = max(y0 - pad, 0)
    x0p = max(x0 - pad, 0)
    z1p = min(z1 + pad, Z)
    y1p = min(y1 + pad, Y)
    x1p = min(x1 + pad, X)

    roi = blob_b[z0p:z1p, y0p:y1p, x0p:x1p]

    # EDT + smooth
    dist = ndi.distance_transform_edt(roi)
    if hmax_sigma and hmax_sigma > 0:
        dist = ndi.gaussian_filter(dist, sigma=float(hmax_sigma))

    # markers from h-maxima
    peaks = (h_maxima(dist, h=float(hmax_h)) & roi)
    markers, n_mark = ndi.label(peaks)
    if debug:
        zmin_g = z0p + z_global_offset
        zmax_g = (z1p - 1) + z_global_offset
        print(
            f"[blob_split]{debug_prefix}hmax: roi_z=[{zmin_g},{zmax_g}] markers={n_mark} h={hmax_h} sigma={hmax_sigma}",
            flush=True,
        )

    if n_mark < int(hmax_min_markers):
        if debug:
            print(f"[blob_split]{debug_prefix}hmax FAIL: markers<{hmax_min_markers}", flush=True)
        return np.zeros_like(blob, dtype=np.uint8)

    # skimage watershed
    conn = get_structure(connectivity)
    labels = watershed(-dist, markers, mask=roi, connectivity=conn)

    # collapse multi-regions -> 2 classes by region z-centroid (global z)
    out_roi = np.zeros_like(labels, dtype=np.uint8)
    max_k = int(labels.max())
    for k in range(1, max_k + 1):
        reg = (labels == k)
        if not reg.any():
            continue
        zz = np.where(reg)[0]
        cz_global = float(zz.mean()) + float(z0p + z_global_offset)
        out_roi[reg] = 1 if cz_global <= thr_z else 2

    # ensure non-destructive: any leftover roi vox -> z fallback
    unl = roi & (out_roi == 0)
    if unl.any():
        z_grid = (np.arange(out_roi.shape[0], dtype=np.float32) + float(z0p + z_global_offset))[:, None, None]
        out_roi[unl & (z_grid <= thr_z)] = 1
        out_roi[unl & (z_grid > thr_z)] = 2

    # paste back
    out = np.zeros_like(blob, dtype=np.uint8)
    out[z0p:z1p, y0p:y1p, x0p:x1p] = out_roi

    # acceptance: must have both sides with enough voxels
    n1 = int((out == 1).sum())
    n2 = int((out == 2).sum())
    if n1 < int(min_side_vox) or n2 < int(min_side_vox):
        if debug:
            print(
                f"[blob_split]{debug_prefix}hmax REJECT: lower={n1} upper={n2} need>={min_side_vox}",
                flush=True,
            )
        return np.zeros_like(blob, dtype=np.uint8)

    if debug:
        print(
            f"[blob_split]{debug_prefix}hmax OK: lower={n1} upper={n2}",
            flush=True,
        )

    return out


def split_blob_by_thickness_watershed(
    blob_mask: np.ndarray,
    z_global_offset: int,
    thr_z: float,
    seed_margin: int,
    seed_dt_min: float,
    seed_dt_percentile: float,
    cost_gamma: float,
    connectivity: int = 26,
    min_seed_vox: int = 50,
    # debug（你原本 repo 有 --debug 的話，這裡吃得到）
    debug: bool = False,
    debug_prefix: str = "",
    # fallback：預設開啟 hmax（安全：切不出兩邊就會回 0，caller 繼續 centroid fallback）
    fallback: str = "hmax",  # "none" | "hmax"
    hmax_h: float = 3.0,
    hmax_sigma: float = 1.0,
    hmax_pad: int = 10,
    hmax_min_markers: int = 2,
):
    """Split a suspicious blob into lower/upper.

    回傳 ROI coords label map：
      1 = lower, 2 = upper, 0 elsewhere

    失敗回傳全 0（caller 可 fallback 到 centroid）。
    """

    blob = (blob_mask > 0).astype(np.uint8, copy=False)
    if blob.sum() == 0:
        return np.zeros_like(blob, dtype=np.uint8)

    if debug_prefix:
        debug_prefix = f"[{debug_prefix}] "

    if debug:
        zz = np.where(blob.any(axis=(1, 2)))[0]
        if zz.size:
            zmin_g = int(zz.min()) + int(z_global_offset)
            zmax_g = int(zz.max()) + int(z_global_offset)
        else:
            zmin_g = zmax_g = int(z_global_offset)
        print(
            f"[blob_split]{debug_prefix}vox={int(blob.sum())} z=[{zmin_g},{zmax_g}] thr={thr_z:.2f} seed_margin={seed_margin} min_seed_vox={min_seed_vox}",
            flush=True,
        )

    # --- seed-based thickness watershed ---
    edt = ndi.distance_transform_edt(blob)
    edt_vals = edt[blob > 0]
    seed_dt = _relax_seed_dt(edt_vals, seed_dt_min, seed_dt_percentile)

    z_roi = np.arange(blob.shape[0], dtype=np.float32) + float(z_global_offset)
    z_grid = z_roi[:, None, None]

    upper_seed = (blob > 0) & (z_grid >= (thr_z + seed_margin)) & (edt >= seed_dt)
    lower_seed = (blob > 0) & (z_grid <= (thr_z - seed_margin)) & (edt >= seed_dt)

    if debug:
        print(
            f"[blob_split]{debug_prefix}seed_dt={seed_dt:.2f} (min={seed_dt_min}, pct={seed_dt_percentile}) upper_seed={int(upper_seed.sum())} lower_seed={int(lower_seed.sum())}",
            flush=True,
        )

    # relax if too few seeds
    if upper_seed.sum() < min_seed_vox or lower_seed.sum() < min_seed_vox:
        for pct in [max(50.0, seed_dt_percentile - 20.0), 40.0, 30.0, 20.0]:
            seed_dt2 = _relax_seed_dt(edt_vals, max(0.5, 0.5 * seed_dt_min), pct)
            upper_seed = (blob > 0) & (z_grid >= (thr_z + seed_margin)) & (edt >= seed_dt2)
            lower_seed = (blob > 0) & (z_grid <= (thr_z - seed_margin)) & (edt >= seed_dt2)
            if debug:
                print(
                    f"[blob_split]{debug_prefix}relax pct={pct:.0f} seed_dt={seed_dt2:.2f} upper_seed={int(upper_seed.sum())} lower_seed={int(lower_seed.sum())}",
                    flush=True,
                )
            if upper_seed.sum() >= min_seed_vox and lower_seed.sum() >= min_seed_vox:
                seed_dt = seed_dt2
                break

    # still not enough -> fallback
    if upper_seed.sum() < min_seed_vox or lower_seed.sum() < min_seed_vox:
        if debug:
            print(
                f"[blob_split]{debug_prefix}FAIL seeds: upper={int(upper_seed.sum())} lower={int(lower_seed.sum())} need>={min_seed_vox}",
                flush=True,
            )

        if str(fallback).lower() == "hmax":
            return _split_blob_hmax_fallback(
                blob,
                z_global_offset=z_global_offset,
                thr_z=thr_z,
                connectivity=connectivity,
                hmax_h=hmax_h,
                hmax_sigma=hmax_sigma,
                hmax_pad=hmax_pad,
                hmax_min_markers=hmax_min_markers,
                min_side_vox=min_seed_vox,
                debug=debug,
                debug_prefix=debug_prefix,
            )

        return np.zeros_like(blob, dtype=np.uint8)

    if debug:
        print(
            f"[blob_split]{debug_prefix}OK: using seed_dt={seed_dt:.2f} upper_seed={int(upper_seed.sum())} lower_seed={int(lower_seed.sum())}",
            flush=True,
        )

    markers = np.zeros(blob.shape, dtype=np.int32)
    markers[lower_seed] = 1
    markers[upper_seed] = 2

    cost = make_cost_from_edt(edt, gamma=cost_gamma)
    cost2 = cost.copy()
    cost2[blob == 0] = np.uint16(65535)  # prevent leaking outside blob

    struct = get_structure(connectivity)
    lab = ndi.watershed_ift(cost2, markers=markers, structure=struct).astype(np.uint8)

    out = np.zeros_like(blob, dtype=np.uint8)
    out[blob > 0] = lab[blob > 0]

    # very rare: unlabeled -> z fallback (still non-destructive)
    unl = (blob > 0) & (out == 0)
    if unl.any():
        out[unl & (z_grid <= thr_z)] = 1
        out[unl & (z_grid > thr_z)] = 2

    return out