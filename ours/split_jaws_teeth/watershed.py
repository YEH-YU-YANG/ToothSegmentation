#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from scipy import ndimage as ndi

from split_jaws.io_utils import load_mask_npy, save_mask_npy, ensure_binary_uint8  # :contentReference[oaicite:4]{index=4}
from split_jaws.cc_utils import get_structure  # :contentReference[oaicite:5]{index=5}
from split_jaws.watershed_cost import make_cost_from_edt  # :contentReference[oaicite:6]{index=6}
from split_jaws.thr_estimation import estimate_thr_from_profile  # :contentReference[oaicite:7]{index=7}

def remove_small_components(mask_u8: np.ndarray, min_vox: int, connectivity: int = 26) -> np.ndarray:
    """Remove connected components smaller than min_vox. mask_u8 is 0/1 uint8."""
    if min_vox <= 0:
        return mask_u8
    m = (mask_u8 > 0).astype(np.uint8, copy=False)
    if int(m.sum()) == 0:
        return m

    struct = get_structure(connectivity)
    lab, n = ndi.label(m, structure=struct)
    if n == 0:
        return m

    sizes = np.bincount(lab.ravel())
    # label 0 is background
    keep = sizes >= int(min_vox)
    keep[0] = False

    cleaned = keep[lab].astype(np.uint8)
    return cleaned

def _relax_seed_dt(edt_vals: np.ndarray, seed_dt_min: float, percentile: float) -> float:
    if edt_vals.size == 0:
        return float(seed_dt_min)
    p = float(np.percentile(edt_vals, percentile))
    return max(float(seed_dt_min), p)


def split_upper_lower_watershed_only(
    mask_u8: np.ndarray,
    thr_z: float,
    seed_margin: int = 10,
    seed_dt_min: float = 2.0,
    seed_dt_percentile: float = 70.0,
    cost_gamma: float = 2.2,
    connectivity: int = 26,
    min_seed_vox: int = 200,
    min_cc_vox: int = 200
):
    """
    Whole-volume 2-marker watershed on binary mask.
    Returns: upper_u8, lower_u8 (both 0/1 uint8).
    """
    teeth = (mask_u8 > 0).astype(np.uint8, copy=False)
    teeth = remove_small_components(teeth, min_vox=min_cc_vox, connectivity=connectivity)

    if int(teeth.sum()) == 0:
        z, y, x = teeth.shape
        return np.zeros((z, y, x), np.uint8), np.zeros((z, y, x), np.uint8)

    # Thickness proxy
    edt = ndi.distance_transform_edt(teeth)
    edt_vals = edt[teeth > 0]

    # Choose seed thickness threshold
    seed_dt = _relax_seed_dt(edt_vals, seed_dt_min, seed_dt_percentile)

    Z = teeth.shape[0]
    z_grid = np.arange(Z, dtype=np.float32)[:, None, None]

    # Initial seeds: thick-core & far from thr plane
    upper_seed = (teeth > 0) & (z_grid >= (thr_z + seed_margin)) & (edt >= seed_dt)
    lower_seed = (teeth > 0) & (z_grid <= (thr_z - seed_margin)) & (edt >= seed_dt)

    # Relax strategy if seeds are too few
    if int(upper_seed.sum()) < min_seed_vox or int(lower_seed.sum()) < min_seed_vox:
        pct_list = [
            max(50.0, seed_dt_percentile - 20.0),
            40.0, 30.0, 20.0, 10.0
        ]
        for pct in pct_list:
            seed_dt2 = _relax_seed_dt(edt_vals, max(0.5, 0.5 * seed_dt_min), pct)
            upper_seed2 = (teeth > 0) & (z_grid >= (thr_z + max(1, seed_margin // 2))) & (edt >= seed_dt2)
            lower_seed2 = (teeth > 0) & (z_grid <= (thr_z - max(1, seed_margin // 2))) & (edt >= seed_dt2)
            if int(upper_seed2.sum()) >= min_seed_vox and int(lower_seed2.sum()) >= min_seed_vox:
                upper_seed, lower_seed = upper_seed2, lower_seed2
                seed_dt = seed_dt2
                break

    # If still fail: fallback to z-split (避免全空)
    if int(upper_seed.sum()) < min_seed_vox or int(lower_seed.sum()) < min_seed_vox:
        lower = ((teeth > 0) & (z_grid <= thr_z)).astype(np.uint8)
        upper = ((teeth > 0) & (z_grid > thr_z)).astype(np.uint8)
        return upper, lower

    # Markers: 1=lower, 2=upper
    markers = np.zeros(teeth.shape, dtype=np.int32)
    markers[lower_seed] = 1
    markers[upper_seed] = 2

    # Cost: thin -> high cost, thick -> low cost
    cost = make_cost_from_edt(edt, gamma=cost_gamma)
    cost2 = cost.copy()
    cost2[teeth == 0] = np.uint16(65535)  # no leaking outside mask

    struct = get_structure(connectivity)
    lab = ndi.watershed_ift(cost2, markers=markers, structure=struct).astype(np.uint8)

    # Extract
    lower = ((lab == 1) & (teeth > 0)).astype(np.uint8)
    upper = ((lab == 2) & (teeth > 0)).astype(np.uint8)

    # Fill unlabeled (rare) by z-side
    unl = (teeth > 0) & (lab == 0)
    if bool(unl.any()):
        lower[unl & (z_grid <= thr_z)] = 1
        upper[unl & (z_grid > thr_z)] = 1

    # Resolve overlaps (should be rare)
    ov = (upper > 0) & (lower > 0)
    if bool(ov.any()):
        upper[ov & (z_grid <= thr_z)] = 0
        lower[ov & (z_grid > thr_z)] = 0
        
    upper = remove_small_components(upper, min_vox=min_cc_vox, connectivity=connectivity)
    lower = remove_small_components(lower, min_vox=min_cc_vox, connectivity=connectivity)

    return upper, lower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_dir", required=True)
    ap.add_argument("--input", default="mask_binary.npy")
    ap.add_argument("--out_upper", default="teeth_upper.npy")
    ap.add_argument("--out_lower", default="teeth_lower.npy")

    ap.add_argument("--thr_mode", choices=["mid", "profile", "override"], default="profile")
    ap.add_argument("--thr_override", type=float, default=None)

    ap.add_argument("--seed_margin", type=int, default=10)
    ap.add_argument("--seed_dt_min", type=float, default=2.0)
    ap.add_argument("--seed_dt_percentile", type=float, default=70.0)
    ap.add_argument("--cost_gamma", type=float, default=2.2)
    ap.add_argument("--connectivity", type=int, default=26)
    ap.add_argument("--min_seed_vox", type=int, default=200)
    ap.add_argument("--min_cc_vox", type=int, default=200)
    ap.add_argument("--swap", action="store_true")
    args = ap.parse_args()

    in_path = os.path.join(args.patient_dir, args.input)
    mask = ensure_binary_uint8(load_mask_npy(in_path))

    # thr
    if args.thr_mode == "override":
        if args.thr_override is None:
            raise ValueError("--thr_mode override needs --thr_override")
        thr_z = float(args.thr_override)
    elif args.thr_mode == "mid":
        zz = np.nonzero(mask > 0)[0]
        thr_z = float(0.5 * (float(zz.min()) + float(zz.max()))) if zz.size else float(mask.shape[0] // 2)
    else:  # profile
        thr_z = float(estimate_thr_from_profile(mask))

    upper, lower = split_upper_lower_watershed_only(
        mask_u8=mask,
        thr_z=thr_z,
        seed_margin=args.seed_margin,
        seed_dt_min=args.seed_dt_min,
        seed_dt_percentile=args.seed_dt_percentile,
        cost_gamma=args.cost_gamma,
        connectivity=args.connectivity,
        min_seed_vox=args.min_seed_vox,
    )

    if args.swap:
        upper, lower = lower, upper

    save_mask_npy(os.path.join(args.patient_dir, args.out_upper), upper.astype(np.uint8))
    save_mask_npy(os.path.join(args.patient_dir, args.out_lower), lower.astype(np.uint8))

    print(f"thr_z={thr_z:.2f}")
    print(f"upper vox={int(upper.sum())}, lower vox={int(lower.sum())}")


if __name__ == "__main__":
    main()
