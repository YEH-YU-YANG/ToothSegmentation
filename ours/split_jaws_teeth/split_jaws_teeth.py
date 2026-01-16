#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from scipy import ndimage as ndi

from split_jaws.io_utils import load_mask_npy, save_mask_npy, ensure_binary_uint8
from split_jaws.cc_utils import get_structure, remove_small_cc, bbox_pad
from split_jaws.thr_estimation import estimate_thr_from_cc
from split_jaws.blob_split import split_blob_by_thickness_watershed
from split_jaws.refine import band_refine_watershed


def log(msg: str):
    print(msg, flush=True)


def enforce_cc_side(mask, upper, lower, thr_z, connectivity=26, margin=12, debug=False):
    """
    If a CC is clearly away from the split plane, force it to the correct side by z-range.
    - if zmax <= thr - margin -> LOWER
    - if zmin >= thr + margin -> UPPER
    """
    struct = get_structure(connectivity)
    lbl, _ = ndi.label(mask > 0, structure=struct)
    objs = ndi.find_objects(lbl)

    moved = 0
    for cc_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        cc = (lbl[slc] == cc_id)
        if not cc.any():
            continue

        zz = np.nonzero(cc)[0] + slc[0].start
        zmin = int(zz.min())
        zmax = int(zz.max())

        target = None
        if zmax <= (thr_z - margin):
            target = "LOWER"
        elif zmin >= (thr_z + margin):
            target = "UPPER"

        if target is None:
            continue

        u = int(upper[slc][cc].sum())
        l = int(lower[slc][cc].sum())

        if target == "LOWER" and u > 0:
            upper[slc][cc] = 0
            lower[slc][cc] = 1
            moved += 1
            if debug:
                log(f"[enforce] cc{cc_id} z=[{zmin},{zmax}] -> LOWER (was up={u} lo={l})")

        if target == "UPPER" and l > 0:
            lower[slc][cc] = 0
            upper[slc][cc] = 1
            moved += 1
            if debug:
                log(f"[enforce] cc{cc_id} z=[{zmin},{zmax}] -> UPPER (was up={u} lo={l})")

    return upper, lower, moved


def fix_mixed_ccs(mask, upper, lower, thr_z, connectivity=26, keep_margin=12, debug=False):
    """
    Fix CC leakage after refine:
    - If a CC spans across thr±keep_margin -> keep (avoid breaking true fused blobs).
    - Else if CC has voxels in BOTH upper and lower -> assign whole CC to the majority side.
    """
    struct = get_structure(connectivity)
    lbl, _ = ndi.label(mask > 0, structure=struct)
    objs = ndi.find_objects(lbl)

    fixed = 0
    kept = 0

    for cc_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        cc = (lbl[slc] == cc_id)
        if not cc.any():
            continue

        u = int(upper[slc][cc].sum())
        l = int(lower[slc][cc].sum())
        if u == 0 or l == 0:
            continue

        zz = np.nonzero(cc)[0] + slc[0].start
        zmin = int(zz.min())
        zmax = int(zz.max())

        # true cross-plane big blob: keep (do NOT collapse)
        if (zmin <= thr_z - keep_margin) and (zmax >= thr_z + keep_margin):
            kept += 1
            if debug:
                log(f"[mix_keep] cc{cc_id} z=[{zmin},{zmax}] u={u} l={l} (cross thr±{keep_margin})")
            continue

        # leakage: assign by majority
        if l >= u:
            upper[slc][cc] = 0
            lower[slc][cc] = 1
            side = "LOWER"
        else:
            lower[slc][cc] = 0
            upper[slc][cc] = 1
            side = "UPPER"

        fixed += 1
        if debug:
            log(f"[mix_fix]  cc{cc_id} z=[{zmin},{zmax}] u={u} l={l} -> {side}")

    return upper, lower, fixed, kept


def hard_clamp_by_z(mask, upper, lower, thr_z, clamp_margin=3, debug=False):
    """
    HARD constraint:
      z <= thr - clamp_margin  => must be LOWER
      z >= thr + clamp_margin  => must be UPPER
    Only leaves a thin ambiguous band to be decided by blob_split/refine.
    This directly prevents 'lower voxels end up in upper far away from the split plane'.
    """
    if clamp_margin <= 0:
        return upper, lower

    Z = mask.shape[0]
    z = np.arange(Z, dtype=np.float32)[:, None, None]
    m = (mask > 0)

    lower_zone = m & (z <= (thr_z - clamp_margin))
    upper_zone = m & (z >= (thr_z + clamp_margin))

    # force
    upper[lower_zone] = 0
    lower[lower_zone] = 1
    lower[upper_zone] = 0
    upper[upper_zone] = 1

    if debug:
        log(f"[clamp] clamp_margin={clamp_margin} forced_lower={int(lower_zone.sum())} forced_upper={int(upper_zone.sum())}")

    return upper, lower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_dir", required=True)
    ap.add_argument("--input", default="mask_binary.npy")
    ap.add_argument("--out_upper", default="teeth_upper.npy")
    ap.add_argument("--out_lower", default="teeth_lower.npy")

    # global clean
    ap.add_argument("--min_global", type=int, default=13000)

    # CC stats
    ap.add_argument("--connectivity", type=int, default=26)

    # thr estimate
    ap.add_argument("--min_cc_for_stats", type=int, default=500)
    ap.add_argument("--thr_override", type=float, default=None)

    # suspicious blob detection
    ap.add_argument("--span_margin", type=int, default=6)
    ap.add_argument("--cc_min_vox_susp", type=int, default=3000)
    ap.add_argument("--roi_pad", type=int, default=6)

    # blob split
    ap.add_argument("--seed_margin", type=int, default=10)
    ap.add_argument("--seed_dt_min", type=float, default=2.0)
    ap.add_argument("--seed_dt_percentile", type=float, default=70.0)
    ap.add_argument("--cost_gamma", type=float, default=2.2)
    ap.add_argument("--min_seed_vox", type=int, default=80)

    # refine
    ap.add_argument("--refine_band", type=int, default=18)
    ap.add_argument("--refine_min_seed_vox", type=int, default=200)

    # protections
    ap.add_argument("--enforce_cc_side", action="store_true")
    ap.add_argument("--enforce_margin", type=int, default=12)

    ap.add_argument("--fix_mixed_cc", action="store_true")
    ap.add_argument("--keep_margin", type=int, default=12)

    ap.add_argument("--hard_clamp_margin", type=int, default=0,
                    help=">0: hard clamp by z outside thr±margin (prevents cross-label far from split plane)")

    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    in_path = os.path.join(args.patient_dir, args.input)
    mask = ensure_binary_uint8(load_mask_npy(in_path))

    log(f"mask shape: {mask.shape} dtype: {mask.dtype}")
    log(f"initial teeth voxels: {int(mask.sum())}")

    mask = remove_small_cc(mask, min_vox=args.min_global, connectivity=args.connectivity)
    log(f"after global clean voxels: {int(mask.sum())}")

    # thr
    if args.thr_override is not None:
        thr_z = float(args.thr_override)
        info = {"mode": "override"}
    else:
        thr_z, info = estimate_thr_from_cc(mask, connectivity=args.connectivity, min_cc_for_stats=args.min_cc_for_stats)

    if info.get("mode") == "kmeans":
        c_low, c_high = info["centers"]
        log(f"[cc_kmeans] centers=({c_low:.2f},{c_high:.2f}), thr={thr_z:.2f}")
    else:
        log(f"[thr] mode={info.get('mode')}, thr={thr_z:.2f}")

    struct = get_structure(args.connectivity)
    lbl, n = ndi.label(mask > 0, structure=struct)
    log(f"[cc] num_cc={n}")

    upper = np.zeros_like(mask, dtype=np.uint8)
    lower = np.zeros_like(mask, dtype=np.uint8)

    objs = ndi.find_objects(lbl)
    suspicious_cnt = 0
    blob_split_cnt = 0

    for cc_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue

        cc = (lbl[slc] == cc_id)
        vox = int(cc.sum())
        if vox == 0:
            continue

        zz = np.nonzero(cc)[0] + slc[0].start
        zmin = int(zz.min())
        zmax = int(zz.max())
        cz = float(zz.mean())

        is_susp = (zmin <= (thr_z - args.span_margin)) and (zmax >= (thr_z + args.span_margin)) and (vox >= args.cc_min_vox_susp)

        if not is_susp:
            if cz <= thr_z:
                lower[slc][cc] = 1
            else:
                upper[slc][cc] = 1
            continue

        suspicious_cnt += 1
        if args.debug:
            log(f"[dbg_cc] id={cc_id} vox={vox} zmin={zmin} zmax={zmax} cz={cz:.2f} thr={thr_z:.2f}")

        roi = bbox_pad(slc, mask.shape, args.roi_pad)
        blob = (lbl[roi] == cc_id).astype(np.uint8)

        lab = split_blob_by_thickness_watershed(
            blob_mask=blob,
            z_global_offset=roi[0].start,
            thr_z=thr_z,
            seed_margin=args.seed_margin,
            seed_dt_min=args.seed_dt_min,
            seed_dt_percentile=args.seed_dt_percentile,
            cost_gamma=args.cost_gamma,
            connectivity=args.connectivity,
            min_seed_vox=args.min_seed_vox,
            debug=args.debug,
            debug_prefix=f"cc{cc_id}",
        )

        if lab.sum() == 0:
            if args.debug:
                log(f"[dbg_cc] cc{cc_id} blob_split=FAIL -> centroid fallback (cz={cz:.2f}, thr={thr_z:.2f})")
            if cz <= thr_z:
                lower[slc][cc] = 1
            else:
                upper[slc][cc] = 1
            continue

        blob_split_cnt += 1
        lower[roi][lab == 1] = 1
        upper[roi][lab == 2] = 1

    log(f"[suspicious] detected={suspicious_cnt}, blob_split_applied={blob_split_cnt}")

    # refine (你的 refine.py 已修 ROI)
    upper, lower = band_refine_watershed(
        teeth_mask=mask,
        upper=upper,
        lower=lower,
        thr_z=thr_z,
        refine_band=args.refine_band,
        seed_margin=args.seed_margin,
        seed_dt_min=args.seed_dt_min,
        seed_dt_percentile=args.seed_dt_percentile,
        cost_gamma=args.cost_gamma,
        connectivity=args.connectivity,
        min_seed_vox=args.refine_min_seed_vox,
        debug=args.debug,
    )

    # safety inside mask
    upper = ((upper > 0) & (mask > 0)).astype(np.uint8)
    lower = ((lower > 0) & (mask > 0)).astype(np.uint8)

    # resolve overlap by z
    overlap = (upper > 0) & (lower > 0)
    if overlap.any():
        z = np.arange(mask.shape[0], dtype=np.float32)[:, None, None]
        take_lower = overlap & (z <= thr_z)
        take_upper = overlap & (z > thr_z)
        upper[take_lower] = 0
        lower[take_upper] = 0

    # HARD clamp (最直接治“下排跑上排”)
    if args.hard_clamp_margin > 0:
        upper, lower = hard_clamp_by_z(mask, upper, lower, thr_z, clamp_margin=args.hard_clamp_margin, debug=args.debug)

    # CC-level enforcement (治“整顆牙/整串牙跑錯邊”)
    if args.enforce_cc_side:
        upper, lower, moved = enforce_cc_side(
            mask=mask, upper=upper, lower=lower,
            thr_z=thr_z, connectivity=args.connectivity,
            margin=args.enforce_margin, debug=args.debug
        )
        if args.debug:
            log(f"[enforce] moved_cc={moved}")

    # leakage fix (治 refine 的 CC 混邊)
    if args.fix_mixed_cc:
        upper, lower, fixed, kept = fix_mixed_ccs(
            mask=mask, upper=upper, lower=lower,
            thr_z=thr_z, connectivity=args.connectivity,
            keep_margin=args.keep_margin, debug=args.debug
        )
        if args.debug:
            log(f"[mix_summary] fixed={fixed} kept_crossing={kept}")

    if args.swap:
        upper, lower = lower, upper

    out_upper = os.path.join(args.patient_dir, args.out_upper)
    out_lower = os.path.join(args.patient_dir, args.out_lower)
    save_mask_npy(out_upper, upper.astype(np.uint8))
    save_mask_npy(out_lower, lower.astype(np.uint8))

    log(f"saved: {out_upper} (vox={int(upper.sum())})")
    log(f"saved: {out_lower} (vox={int(lower.sum())})")


if __name__ == "__main__":
    main()
