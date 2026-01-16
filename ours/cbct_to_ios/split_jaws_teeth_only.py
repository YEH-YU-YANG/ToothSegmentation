# split_jaws_teeth_only.py
#
# Auto-first jaw split for a teeth-only binary mask:
# - No hard global plane cut (avoid truncating roots)
# - No destructive voxel deletion by default (interface_gap defaults to 0)
# - Minimal required CLI: --patient_dir/--mask and --min_global
#
# Pipeline:
#  1) Global clean (remove tiny CCs)
#  2) Choose a robust plane (axis=0)  [for reporting / reference only]
#  3) Compute cc_thr by kmeans on CC centroids
#  4) Split component-wise:
#      - normal CC: assign whole CC by centroid z vs cc_thr
#      - suspicious CC spanning both sides:
#          (a) DT-core split (NON-destructive, with thick-core seeds + z-bias)
#          (b) erosion split fallback (NON-destructive reconstruction)
#          (c) band-cut fallback only for extremely large blobs (still non-destructive overall)
#  5) Refine near interface: RELABEL only (no deletion unless you override interface_gap>0)
#  6) Move obviously wrong islands (NON-destructive: move whole small CCs to the other jaw)
#  7) Auto-adjust loop: if misassigned thin-slice remains, increase margins/band/r_max and retry.
#
# Recommended:
#   --interface_gap 0  (default)

import os
import argparse
import numpy as np
from scipy import ndimage


# ----------------------------
# Utils
# ----------------------------
def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return x.astype(np.float32)
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x.astype(np.float32), k, mode="same")


def remove_small_components_3d(bin_mask: np.ndarray, min_voxels: int) -> np.ndarray:
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
    if k is None or k <= 0:
        return bin_mask.astype(bool)
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(bin_mask, structure=structure)
    if num == 0:
        return bin_mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    top = np.argsort(counts)[::-1][: int(k)]
    return np.isin(labeled, top)


def weighted_kmeans_1d(vals: np.ndarray, weights: np.ndarray, iters: int = 80):
    """1D weighted kmeans(k=2). Returns assign(0/1), c0, c1 with c0 < c1."""
    vals = vals.astype(np.float32)
    w = weights.astype(np.float32)

    c0, c1 = float(vals.min()), float(vals.max())
    assign = np.zeros(vals.shape[0], dtype=np.int32)

    for _ in range(iters):
        d0 = np.abs(vals - c0)
        d1 = np.abs(vals - c1)
        new_assign = (d1 < d0).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign

        for k in (0, 1):
            m = assign == k
            if not np.any(m):
                continue
            ww = w[m]
            vv = vals[m]
            s = float(ww.sum())
            if s > 0:
                if k == 0:
                    c0 = float((vv * ww).sum() / (s + 1e-8))
                else:
                    c1 = float((vv * ww).sum() / (s + 1e-8))

    if c0 > c1:
        c0, c1 = c1, c0
        assign = 1 - assign
    return assign, c0, c1


def enforce_disjoint(upper: np.ndarray, lower: np.ndarray, cc_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Resolve overlap by z vs cc_thr (non-destructive)."""
    upper = upper.astype(bool)
    lower = lower.astype(bool)
    overlap = upper & lower
    if not np.any(overlap):
        return upper, lower

    Z = upper.shape[0]
    z = np.arange(Z)[:, None, None].astype(np.float32)
    thr = float(cc_thr)

    choose_upper = (z < thr)
    upper[overlap] = choose_upper[overlap]
    lower[overlap] = ~choose_upper[overlap]
    return upper, lower


# ----------------------------
# Plane selection (axis0)
# ----------------------------
def axis0_profile(teeth: np.ndarray) -> np.ndarray:
    return teeth.sum(axis=(1, 2)).astype(np.float32)


def find_plane_axis0_by_profile(teeth: np.ndarray, smooth_win: int) -> int:
    prof_s = smooth_1d(axis0_profile(teeth), smooth_win)
    d = len(prof_s)
    mid = d // 2
    p1 = int(np.argmax(prof_s[:mid]))
    p2 = int(np.argmax(prof_s[mid:]) + mid)
    if p1 > p2:
        p1, p2 = p2, p1
    valley = int(np.argmin(prof_s[p1 : p2 + 1]) + p1)
    return valley


def find_plane_axis0_by_kmeans(teeth: np.ndarray):
    coords = np.argwhere(teeth)
    z = coords[:, 0].astype(np.float32)
    _, c0, c1 = weighted_kmeans_1d(z, np.ones_like(z, dtype=np.float32))
    thr = (c0 + c1) / 2.0
    plane = int(np.clip(round(thr), 0, teeth.shape[0] - 1))
    return plane, c0, c1


def eval_plane(teeth: np.ndarray, plane: int):
    plane = int(np.clip(plane, 0, teeth.shape[0] - 1))
    below = int(teeth[:plane].sum())
    above = int(teeth[plane:].sum())
    total = below + above
    min_frac = (min(below, above) / total) if total > 0 else 0.0
    return float(min_frac), below, above, total


def refine_plane_to_local_valley(teeth: np.ndarray, plane0: int, smooth_win: int, radius: int) -> int:
    prof_s = smooth_1d(axis0_profile(teeth), smooth_win)
    d = len(prof_s)
    lo = max(0, int(plane0) - int(radius))
    hi = min(d, int(plane0) + int(radius) + 1)
    if hi <= lo + 1:
        return int(np.clip(plane0, 0, d - 1))
    return int(np.argmin(prof_s[lo:hi]) + lo)


def choose_plane_axis0(teeth: np.ndarray, smooth_win: int, refine_radius: int) -> int:
    plane_prof = find_plane_axis0_by_profile(teeth, smooth_win=smooth_win)
    plane_km, _, _ = find_plane_axis0_by_kmeans(teeth)

    cand = []
    for name, p in [("profile", plane_prof), ("kmeans", plane_km)]:
        mf, below, above, total = eval_plane(teeth, p)
        cand.append((mf, name, int(p), below, above, total))

    cand.sort(
        key=lambda x: (
            x[0],
            -abs((x[3] / (x[5] + 1e-8)) - 0.5),
            -x[2],
        ),
        reverse=True,
    )
    best_mf, best_plane = cand[0][0], cand[0][2]
    if best_mf < 0.02:
        best_plane = plane_km

    best_plane_ref = refine_plane_to_local_valley(teeth, best_plane, smooth_win=smooth_win, radius=refine_radius)
    mf0, *_ = eval_plane(teeth, best_plane)
    mf1, *_ = eval_plane(teeth, best_plane_ref)
    return best_plane_ref if mf1 >= (mf0 - 1e-6) else best_plane


def cc_kmeans_threshold_axis0(teeth: np.ndarray):
    """kmeans on CC centroids (weights=CC sizes) -> cc_thr."""
    teeth = teeth.astype(bool)
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(teeth, structure=structure)
    if num < 2:
        plane_km, c0, c1 = find_plane_axis0_by_kmeans(teeth)
        thr = (c0 + c1) / 2.0
        return float(thr), float(c0), float(c1), int(num)

    counts = np.bincount(labeled.ravel())
    sizes = counts[1:].astype(np.float32)

    idx = list(range(1, num + 1))
    coms = ndimage.center_of_mass(teeth, labeled, idx)
    z = np.array([c[0] for c in coms], dtype=np.float32)

    _, c0, c1 = weighted_kmeans_1d(z, sizes)
    thr = (c0 + c1) / 2.0
    return float(thr), float(c0), float(c1), int(num)


# ----------------------------
# Non-destructive blob split helpers
# ----------------------------
def try_split_blob_by_erosion(sub: np.ndarray, min_sub_vox: int, erosion_max: int, structure=None):
    """Erode until it splits -> use two largest as seeds -> reconstruct by nearest-seed (NON-destructive)."""
    if structure is None:
        structure = ndimage.generate_binary_structure(3, 2)

    if erosion_max is None or erosion_max <= 0:
        return None, None, 0

    sub = sub.astype(bool)
    if int(sub.sum()) == 0:
        return None, None, 0

    min_sub_vox = int(max(0, min_sub_vox))
    erosion_max = int(max(0, erosion_max))

    for it in range(1, erosion_max + 1):
        er = ndimage.binary_erosion(sub, structure=structure, iterations=it, border_value=0)
        if not np.any(er):
            break

        lab, n = ndimage.label(er, structure=structure)
        if n < 2:
            continue

        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]
        keep = [int(i) for i in order if sizes[i] >= min_sub_vox][:2]
        if len(keep) < 2:
            continue

        seed0 = (lab == keep[0])
        seed1 = (lab == keep[1])

        d0 = ndimage.distance_transform_edt(~seed0)
        d1 = ndimage.distance_transform_edt(~seed1)

        part0 = sub & (d0 <= d1)
        part1 = sub & (d1 < d0)

        if int(part0.sum()) >= min_sub_vox and int(part1.sum()) >= min_sub_vox:
            return part0, part1, it

    return None, None, 0


def try_split_blob_by_dt_core(
    sub: np.ndarray,
    plane_local: int,
    seed_margin: int,
    r_max: int,
    min_seed_vox: int,
    min_part_vox: int,
    z_bias_beta: float,
    stable_dt_min: float = 2.0,   # thick-core seed threshold
    structure=None,
):
    """
    DT-core split (NON-destructive), enhanced:
      - seeds use THICK region only: stable = (dt >= stable_dt_min)
      - reconstruction uses z-biased cost to avoid slab-like misassignment near interface
    """
    if structure is None:
        structure = ndimage.generate_binary_structure(3, 2)

    sub = sub.astype(bool)
    if sub.sum() == 0 or sub.shape[0] < 3:
        return None, None, 0

    Z = sub.shape[0]
    plane_local = int(np.clip(int(plane_local), 0, Z - 1))
    seed_margin = int(max(0, seed_margin))
    r_max = int(max(0, r_max))

    # thickness (distance-to-background)
    dt = ndimage.distance_transform_edt(sub)
    dt_max = float(dt.max())
    if dt_max < 2.0:
        return None, None, 0

    zz = np.arange(Z)[:, None, None]
    stable = (dt >= float(stable_dt_min))

    # THICK seeds only (prevents slab-like thin parts from dominating seeds)
    upper_seed = stable & (zz <= plane_local - seed_margin)
    lower_seed = stable & (zz >= plane_local + seed_margin)

    if int(upper_seed.sum()) < int(min_seed_vox) or int(lower_seed.sum()) < int(min_seed_vox):
        return None, None, 0

    best = None
    r_hi = int(min(r_max, int(np.floor(dt_max))))
    for r in range(2, r_hi + 1):
        core = (dt >= float(r))
        if not core.any():
            break

        labc, nc = ndimage.label(core, structure=structure)
        if nc < 2:
            continue

        sizes = np.bincount(labc.ravel())
        sizes[0] = 0

        ids_u = labc[upper_seed]
        ids_l = labc[lower_seed]
        ids_u = ids_u[ids_u > 0]
        ids_l = ids_l[ids_l > 0]
        if ids_u.size == 0 or ids_l.size == 0:
            continue

        id_u = int(np.bincount(ids_u).argmax())
        id_l = int(np.bincount(ids_l).argmax())
        if id_u == 0 or id_l == 0 or id_u == id_l:
            continue
        if sizes[id_u] < min_part_vox or sizes[id_l] < min_part_vox:
            continue

        # keep the largest r that still splits (strongest bridge removal)
        best = (r, id_u, id_l, labc)

    if best is None:
        return None, None, 0

    r_best, id_u, id_l, labc = best
    seed_u = (labc == id_u)
    seed_l = (labc == id_l)

    # reconstruct full partition with z-biased costs
    d_u = ndimage.distance_transform_edt(~seed_u)
    d_l = ndimage.distance_transform_edt(~seed_l)

    zzf = zz.astype(np.float32)
    beta = float(max(0.0, z_bias_beta))
    # penalize assigning far-below voxels to upper and far-above voxels to lower
    cost_u = d_u + beta * np.maximum(0.0, zzf - float(plane_local))
    cost_l = d_l + beta * np.maximum(0.0, float(plane_local) - zzf)

    part_u = sub & (cost_u <= cost_l)
    part_l = sub & (cost_l < cost_u)

    if int(part_u.sum()) < min_part_vox or int(part_l.sum()) < min_part_vox:
        return None, None, 0

    return part_u, part_l, int(r_best)


# ----------------------------
# Component-wise split
# ----------------------------
def split_jaws_axis0_componentwise(
    teeth: np.ndarray,
    cc_thr: float,
    seed_margin: int,
    r_max: int,
    z_bias_beta: float,
    stable_dt_min: float,
    dtcore_try_min_vox: int,
    min_part_vox: int,
    erosion_min_sub_vox: int,
    erosion_max: int,
    # fallback band-cut only for extremely large blobs
    bandcut_enable: bool,
    bandcut_half_thickness: int,
    try_cut_min_vox: int,
    span_min_frac: float,
):
    teeth = teeth.astype(bool)
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(teeth, structure=structure)
    if num == 0:
        return np.zeros_like(teeth, bool), np.zeros_like(teeth, bool)

    objs = ndimage.find_objects(labeled)
    upper = np.zeros_like(teeth, dtype=bool)
    lower = np.zeros_like(teeth, dtype=bool)

    cc_thr = float(cc_thr)
    thr_plane = int(np.clip(int(round(cc_thr)), 0, teeth.shape[0] - 1))

    for lab in range(1, num + 1):
        slc = objs[lab - 1]
        if slc is None:
            continue
        sub = (labeled[slc] == lab)
        if not np.any(sub):
            continue

        total = int(sub.sum())
        z0, z1 = slc[0].start, slc[0].stop

        # robust z-range (thin slice still triggers)
        zz_any = np.any(sub, axis=(1, 2))
        zmin_g = z0 + int(np.argmax(zz_any)) if np.any(zz_any) else z0
        zmax_g = z0 + int(len(zz_any) - 1 - np.argmax(zz_any[::-1])) if np.any(zz_any) else (z1 - 1)

        crosses_by_margin = (zmin_g <= (cc_thr - seed_margin)) and (zmax_g >= (cc_thr + seed_margin))
        plane_local = thr_plane - z0

        # (A) DT-core split first (non-destructive, thick-core seeds + z-bias)
        if crosses_by_margin and total >= dtcore_try_min_vox and r_max > 0:
            pu, pl, _ = try_split_blob_by_dt_core(
                sub,
                plane_local=plane_local,
                seed_margin=seed_margin,
                r_max=r_max,
                min_seed_vox=erosion_min_sub_vox,
                min_part_vox=min_part_vox,
                z_bias_beta=z_bias_beta,
                stable_dt_min=stable_dt_min,
                structure=structure,
            )
            if pu is not None:
                cz_u = float(ndimage.center_of_mass(pu)[0] + z0)
                cz_l = float(ndimage.center_of_mass(pl)[0] + z0)
                if (cz_u < cc_thr) != (cz_l < cc_thr):
                    if cz_u < cc_thr:
                        upper[slc] |= pu
                        lower[slc] |= pl
                    else:
                        upper[slc] |= pl
                        lower[slc] |= pu
                    continue

        # (B) compute spans_both (classic)
        spans_both = False
        if thr_plane > z0 and thr_plane < z1:
            lp = thr_plane - z0
            below = int(sub[:lp].sum())
            above = total - below
            spans_both = (min(below, above) / max(total, 1)) > float(span_min_frac)

        # (C) erosion fallback (still non-destructive)
        if spans_both and total >= try_cut_min_vox and erosion_max > 0:
            p0, p1, _ = try_split_blob_by_erosion(
                sub,
                min_sub_vox=erosion_min_sub_vox,
                erosion_max=erosion_max,
                structure=structure,
            )
            if p0 is not None:
                cz0 = float(ndimage.center_of_mass(p0)[0] + z0)
                cz1 = float(ndimage.center_of_mass(p1)[0] + z0)
                if (cz0 < cc_thr) != (cz1 < cc_thr):
                    if cz0 < cc_thr:
                        upper[slc] |= p0
                        lower[slc] |= p1
                    else:
                        upper[slc] |= p1
                        lower[slc] |= p0
                    continue

        # (D) band-cut ONLY for extremely large blobs (reassign band by z; non-destructive overall)
        if bandcut_enable and spans_both and total >= try_cut_min_vox and bandcut_half_thickness > 0:
            sub_cut = sub.copy()
            cut_plane_g = int(np.clip(thr_plane, z0, z1 - 1))
            lo = max(0, (cut_plane_g - bandcut_half_thickness) - z0)
            hi = min(z1 - z0, (cut_plane_g + bandcut_half_thickness + 1) - z0)
            if hi > lo:
                sub_cut[lo:hi] = False

            sub_labeled, sub_num = ndimage.label(sub_cut, structure=structure)
            if sub_num >= 2:
                sizes = np.bincount(sub_labeled.ravel())
                sizes[0] = 0
                keep = [i for i in range(1, sub_num + 1) if sizes[i] >= min_part_vox]
                if len(keep) >= 2:
                    for sid in keep:
                        m = (sub_labeled == sid)
                        cz = float(ndimage.center_of_mass(m)[0] + z0)
                        if cz < cc_thr:
                            upper[slc] |= m
                        else:
                            lower[slc] |= m
                    band = sub & (~sub_cut)
                    if np.any(band):
                        z_idx_g = np.arange(z0, z1)[:, None, None].astype(np.float32)
                        upper[slc] |= band & (z_idx_g < cc_thr)
                        lower[slc] |= band & (z_idx_g >= cc_thr)
                    continue

        # Normal CC: assign WHOLE CC by centroid
        cz = float(ndimage.center_of_mass(sub)[0] + z0)
        if cz < cc_thr:
            upper[slc] |= sub
        else:
            lower[slc] |= sub

    return upper, lower


# ----------------------------
# Refinement (relabel only by default)
# ----------------------------
def refine_interface_by_distance(
    teeth: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    cc_thr: float,
    refine_band: int,
    seed_margin: int,
    interface_gap: float,
    roi_pad: int = 6,
):
    teeth = teeth.astype(bool)
    upper = upper.astype(bool)
    lower = lower.astype(bool)

    if refine_band is None or refine_band <= 0:
        return upper, lower

    Z = teeth.shape[0]
    thr = float(cc_thr)

    refine_band = int(max(0, refine_band))
    seed_margin = int(max(0, seed_margin))
    roi_pad = int(max(0, roi_pad))

    half = refine_band + seed_margin + roi_pad
    z_lo = int(max(0, np.floor(thr - half)))
    z_hi = int(min(Z, np.ceil(thr + half) + 1))

    teeth_r = teeth[z_lo:z_hi]
    upper_r = upper[z_lo:z_hi]
    lower_r = lower[z_lo:z_hi]

    z_g = np.arange(z_lo, z_hi, dtype=np.float32)[:, None, None]
    band = teeth_r & (np.abs(z_g - thr) <= float(refine_band))

    upper_seed = upper_r & (z_g < (thr - float(seed_margin)))
    lower_seed = lower_r & (z_g > (thr + float(seed_margin)))

    if upper_seed.sum() < 1000:
        upper_seed = upper_r & (~band)
    if lower_seed.sum() < 1000:
        lower_seed = lower_r & (~band)
    if upper_seed.sum() == 0 or lower_seed.sum() == 0:
        return upper, lower

    du = ndimage.distance_transform_edt(~upper_seed)
    dl = ndimage.distance_transform_edt(~lower_seed)

    assign_upper = du < dl
    ties = (du == dl)
    if np.any(ties):
        tz, ty, tx = np.nonzero(ties)
        assign_upper[tz, ty, tx] = ((tz.astype(np.float32) + float(z_lo)) < float(thr))

    new_upper_r = upper_r.copy()
    new_lower_r = lower_r.copy()
    new_upper_r[band] = assign_upper[band]
    new_lower_r[band] = (~assign_upper)[band]

    # WARNING: deletion can damage crowns
    if interface_gap is not None and interface_gap > 0:
        interface = band & (np.abs(du - dl) <= float(interface_gap))
        new_upper_r[interface] = False
        new_lower_r[interface] = False

    upper2 = upper.copy()
    lower2 = lower.copy()
    upper2[z_lo:z_hi] = new_upper_r
    lower2[z_lo:z_hi] = new_lower_r

    upper2 &= teeth
    lower2 &= teeth
    upper2, lower2 = enforce_disjoint(upper2, lower2, cc_thr=cc_thr)
    return upper2, lower2


# ----------------------------
# Move obviously wrong islands (NON-destructive)
# ----------------------------
def move_obviously_wrong_islands(
    upper: np.ndarray,
    lower: np.ndarray,
    cc_thr: float,
    margin: int = 8,
    max_vox: int = 60000,
):
    """
    Move small/medium CC islands that are clearly on the wrong side of cc_thr.
    No voxel deletion: we move whole CC.
    """
    structure = ndimage.generate_binary_structure(3, 2)
    upper = upper.astype(bool).copy()
    lower = lower.astype(bool).copy()
    thr = float(cc_thr)
    margin = int(max(1, margin))
    max_vox = int(max(100, max_vox))

    def process(src, dst, want_upper: bool):
        lab, n = ndimage.label(src, structure=structure)
        if n == 0:
            return src, dst
        counts = np.bincount(lab.ravel())
        for i in range(1, n + 1):
            vox = int(counts[i])
            if vox == 0 or vox > max_vox:
                continue
            m = (lab == i)
            cz = float(ndimage.center_of_mass(m)[0])
            if want_upper:
                # upper island but clearly too low
                if cz > (thr + margin):
                    src[m] = False
                    dst[m] = True
            else:
                # lower island but clearly too high
                if cz < (thr - margin):
                    src[m] = False
                    dst[m] = True
        return src, dst

    upper, lower = process(upper, lower, want_upper=True)
    lower, upper = process(lower, upper, want_upper=False)

    upper, lower = enforce_disjoint(upper, lower, cc_thr=cc_thr)
    return upper, lower


# ----------------------------
# Auto params + auto check
# ----------------------------
def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(x))))


def clamp_float(x, lo, hi):
    return float(max(lo, min(hi, float(x))))


def compute_auto_params(teeth: np.ndarray, cc_c0: float, cc_c1: float):
    """
    Auto-pick parameters based on centroid separation and volume scale.
    """
    total = int(teeth.sum())
    gap = float(abs(cc_c1 - cc_c0))  # separation in z slices

    # margins/bands scale with centroid gap
    seed_margin = clamp_int(round(0.25 * gap), 6, 18)
    refine_band = clamp_int(round(0.35 * gap), 10, 30)

    # dt-core r_max: more aggressive for bigger separation
    r_max = clamp_int(round(0.45 * gap), 10, 32)

    # z-bias beta: increase with gap but clamp
    z_bias_beta = clamp_float(0.01 * gap, 0.25, 0.65)

    # stable seed thickness threshold (dt>=2 usually enough; can raise to 3 if slabs dominate)
    stable_dt_min = 2.0

    # size thresholds scale with total voxels
    dtcore_try_min_vox = max(3500, int(0.0045 * total))
    min_part_vox = max(1500, int(0.0012 * total))
    erosion_min_sub_vox = max(1200, int(0.0009 * total))

    # big blob fallback thresholds
    try_cut_min_vox = max(20000, int(0.05 * total))
    span_min_frac = 0.08
    bandcut_enable = True
    bandcut_half_thickness = 4

    # island moving parameters
    island_margin = max(8, seed_margin)          # how far from thr counts as "obviously wrong"
    island_max_vox = max(60000, int(0.015 * total))

    return dict(
        seed_margin=seed_margin,
        refine_band=refine_band,
        r_max=r_max,
        z_bias_beta=z_bias_beta,
        stable_dt_min=stable_dt_min,
        dtcore_try_min_vox=dtcore_try_min_vox,
        min_part_vox=min_part_vox,
        erosion_min_sub_vox=erosion_min_sub_vox,
        try_cut_min_vox=try_cut_min_vox,
        span_min_frac=span_min_frac,
        bandcut_enable=bandcut_enable,
        bandcut_half_thickness=bandcut_half_thickness,
        island_margin=island_margin,
        island_max_vox=island_max_vox,
    )


def misassignment_stats(upper: np.ndarray, lower: np.ndarray, cc_thr: float, margin: int):
    """
    bad_upper: upper voxels far below threshold (z > thr + margin)
    bad_lower: lower voxels far above threshold (z < thr - margin)
    """
    Z = upper.shape[0]
    z = np.arange(Z)[:, None, None].astype(np.float32)
    thr = float(cc_thr)
    m = float(max(1, margin))

    bad_upper = upper & (z > (thr + m))
    bad_lower = lower & (z < (thr - m))
    return int(bad_upper.sum()), int(bad_lower.sum())


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--patient_dir", type=str, default=None)
    ap.add_argument("--mask", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)

    # required (you always know this)
    ap.add_argument("--min_global", type=int, required=True)

    # optional
    ap.add_argument("--keep_top_k_global", type=int, default=None)
    ap.add_argument("--min_per_jaw", type=int, default=None)
    ap.add_argument("--keep_top_k_per_jaw", type=int, default=None)

    # plane (mostly for logging)
    ap.add_argument("--plane", type=int, default=None)
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--refine_radius", type=int, default=None)

    # advanced overrides (all default None => auto)
    ap.add_argument("--seed_margin", type=int, default=None)
    ap.add_argument("--refine_band", type=int, default=None)
    ap.add_argument("--erosion_max", type=int, default=None)         # used as r_max in DT-core + erosion fallback
    ap.add_argument("--erosion_min_sub_vox", type=int, default=None)
    ap.add_argument("--dtcore_try_min_vox", type=int, default=None)
    ap.add_argument("--min_part_vox", type=int, default=None)

    ap.add_argument("--try_cut_min_vox", type=int, default=None)
    ap.add_argument("--ambig_frac", type=float, default=None)
    ap.add_argument("--bandcut_half_thickness", type=int, default=None)
    ap.add_argument("--bandcut_disable", action="store_true")

    # DT-core enhancements override
    ap.add_argument("--z_bias_beta", type=float, default=None)
    ap.add_argument("--stable_dt_min", type=float, default=None)

    # island moving override
    ap.add_argument("--island_margin", type=int, default=None)
    ap.add_argument("--island_max_vox", type=int, default=None)

    # do NOT delete voxels by default
    ap.add_argument("--interface_gap", type=float, default=0.0)

    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--auto_rounds", type=int, default=3)

    args = ap.parse_args()

    if args.mask is None and args.patient_dir is None:
        raise ValueError("請提供 --patient_dir 或 --mask")

    if args.mask is not None:
        mask_path = args.mask
        out_dir = args.out_dir if args.out_dir else os.path.dirname(mask_path)
    else:
        mask_path = os.path.join(args.patient_dir, "mask_binary.npy")
        out_dir = args.out_dir if args.out_dir else args.patient_dir

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"找不到 mask：{mask_path}")
    os.makedirs(out_dir, exist_ok=True)

    mask = np.load(mask_path)
    teeth = (mask > 0)

    print("mask shape:", mask.shape, "dtype:", mask.dtype)
    print("initial teeth voxels:", int(teeth.sum()))

    # global clean
    teeth = remove_small_components_3d(teeth, args.min_global)
    teeth = keep_top_k_components_3d(teeth, args.keep_top_k_global)
    total_after = int(teeth.sum())
    print("after global clean voxels:", total_after)

    # plane (only for logging)
    smooth = 21 if args.smooth is None else int(args.smooth)
    refine_radius = 30 if args.refine_radius is None else int(args.refine_radius)

    if args.plane is not None:
        plane = int(np.clip(int(args.plane), 0, teeth.shape[0] - 1))
        print("[plane] manual:", plane)
    else:
        plane = choose_plane_axis0(teeth, smooth_win=smooth, refine_radius=refine_radius)
        print("[plane] auto:", plane)

    mf, below, above, total = eval_plane(teeth, plane)
    print(f"[plane] voxels below={below}, above={above}, total={total}, min_side_frac={mf:.3f}")

    # cc_thr
    cc_thr, cc_c0, cc_c1, cc_num = cc_kmeans_threshold_axis0(teeth)
    print(f"[cc_kmeans] num_cc={cc_num}, centers=({cc_c0:.2f},{cc_c1:.2f}), thr={cc_thr:.2f}")

    # auto params
    auto = compute_auto_params(teeth, cc_c0, cc_c1)

    # merge: user override wins
    seed_margin = auto["seed_margin"] if args.seed_margin is None else int(args.seed_margin)
    refine_band = auto["refine_band"] if args.refine_band is None else int(args.refine_band)
    r_max = auto["r_max"] if args.erosion_max is None else int(args.erosion_max)

    erosion_min_sub_vox = auto["erosion_min_sub_vox"] if args.erosion_min_sub_vox is None else int(args.erosion_min_sub_vox)
    dtcore_try_min_vox = auto["dtcore_try_min_vox"] if args.dtcore_try_min_vox is None else int(args.dtcore_try_min_vox)
    min_part_vox = auto["min_part_vox"] if args.min_part_vox is None else int(args.min_part_vox)

    try_cut_min_vox = auto["try_cut_min_vox"] if args.try_cut_min_vox is None else int(args.try_cut_min_vox)
    span_min_frac = auto["span_min_frac"] if args.ambig_frac is None else float(args.ambig_frac)

    bandcut_enable = (not args.bandcut_disable) and auto["bandcut_enable"]
    bandcut_half_thickness = auto["bandcut_half_thickness"] if args.bandcut_half_thickness is None else int(args.bandcut_half_thickness)

    z_bias_beta = auto["z_bias_beta"] if args.z_bias_beta is None else float(args.z_bias_beta)
    stable_dt_min = auto["stable_dt_min"] if args.stable_dt_min is None else float(args.stable_dt_min)

    island_margin = auto["island_margin"] if args.island_margin is None else int(args.island_margin)
    island_max_vox = auto["island_max_vox"] if args.island_max_vox is None else int(args.island_max_vox)

    # per-jaw clean threshold auto (keep gentle)
    min_per_jaw = 1000 if args.min_per_jaw is None else int(args.min_per_jaw)

    if args.interface_gap > 0:
        print("[WARN] interface_gap > 0 will DELETE voxels and may damage crown surfaces.")

    print("[auto_params]",
          "seed_margin=", seed_margin,
          "refine_band=", refine_band,
          "r_max=", r_max,
          "z_bias_beta=", f"{z_bias_beta:.3f}",
          "stable_dt_min=", f"{stable_dt_min:.2f}",
          "dtcore_try_min_vox=", dtcore_try_min_vox,
          "min_part_vox=", min_part_vox,
          "erosion_min_sub_vox=", erosion_min_sub_vox,
          "try_cut_min_vox=", try_cut_min_vox,
          "span_min_frac=", f"{span_min_frac:.3f}",
          "bandcut=", bandcut_enable,
          "bandcut_half_thickness=", bandcut_half_thickness,
          "island_margin=", island_margin,
          "island_max_vox=", island_max_vox,
          "min_per_jaw=", min_per_jaw)

    # auto-adjust loop
    upper = lower = None
    for round_i in range(0, max(1, int(args.auto_rounds))):
        if round_i > 0:
            # progressively strengthen only when needed
            seed_margin = min(22, seed_margin + 2)
            refine_band = min(36, refine_band + 4)
            r_max = min(36, r_max + 4)
            z_bias_beta = min(0.80, z_bias_beta + 0.08)
            # if slabs still dominate, slightly raise stable_dt_min
            stable_dt_min = min(3.0, stable_dt_min + 0.25)

            island_margin = min(26, island_margin + 2)

            print(f"[auto_adjust] round={round_i} -> "
                  f"seed_margin={seed_margin}, refine_band={refine_band}, r_max={r_max}, "
                  f"z_bias_beta={z_bias_beta:.3f}, stable_dt_min={stable_dt_min:.2f}, island_margin={island_margin}")

        upper, lower = split_jaws_axis0_componentwise(
            teeth=teeth,
            cc_thr=cc_thr,
            seed_margin=seed_margin,
            r_max=r_max,
            z_bias_beta=z_bias_beta,
            stable_dt_min=stable_dt_min,
            dtcore_try_min_vox=dtcore_try_min_vox,
            min_part_vox=min_part_vox,
            erosion_min_sub_vox=erosion_min_sub_vox,
            erosion_max=r_max,
            bandcut_enable=bandcut_enable,
            bandcut_half_thickness=bandcut_half_thickness,
            try_cut_min_vox=try_cut_min_vox,
            span_min_frac=span_min_frac,
        )

        # refine (relabel only)
        if refine_band > 0:
            upper, lower = refine_interface_by_distance(
                teeth, upper, lower,
                cc_thr=cc_thr,
                refine_band=refine_band,
                seed_margin=seed_margin,
                interface_gap=float(args.interface_gap),
                roi_pad=6,
            )

        # move obvious wrong islands (non-destructive)
        upper, lower = move_obviously_wrong_islands(
            upper, lower,
            cc_thr=cc_thr,
            margin=island_margin,
            max_vox=island_max_vox,
        )

        upper, lower = enforce_disjoint(upper, lower, cc_thr=cc_thr)

        # evaluate misassignment
        bad_u, bad_l = misassignment_stats(upper, lower, cc_thr=cc_thr, margin=max(4, seed_margin // 2))
        bad_total = bad_u + bad_l
        print(f"[check] round={round_i} bad_upper={bad_u} bad_lower={bad_l} bad_total={bad_total}")

        if bad_total <= max(800, int(0.001 * total_after)):
            break

    # per-jaw gentle clean
    upper = remove_small_components_3d(upper, min_per_jaw)
    lower = remove_small_components_3d(lower, min_per_jaw)
    upper = keep_top_k_components_3d(upper, args.keep_top_k_per_jaw)
    lower = keep_top_k_components_3d(lower, args.keep_top_k_per_jaw)

    upper, lower = enforce_disjoint(upper, lower, cc_thr=cc_thr)

    if args.swap:
        upper, lower = lower, upper
        print(">> swapped upper/lower")

    print("upper voxels:", int(upper.sum()))
    print("lower voxels:", int(lower.sum()))

    np.save(os.path.join(out_dir, "teeth_clean.npy"), teeth.astype(np.uint8))
    np.save(os.path.join(out_dir, "teeth_upper.npy"), upper.astype(np.uint8))
    np.save(os.path.join(out_dir, "teeth_lower.npy"), lower.astype(np.uint8))

    print("saved:",
          os.path.join(out_dir, "teeth_clean.npy"),
          os.path.join(out_dir, "teeth_upper.npy"),
          os.path.join(out_dir, "teeth_lower.npy"))


if __name__ == "__main__":
    main()
