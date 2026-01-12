# split_jaws_teeth_only.py
#
# Goal: split a teeth-only binary mask into upper/lower jaws WITHOUT cutting off roots
# and WITHOUT destructive voxel deletion at crowns.
#
# Strategy:
#  1) Global clean (remove tiny CCs)
#  2) Choose a robust plane (axis=0)
#  3) Compute cc_thr (jaw separation threshold) by kmeans on CC centroids
#  4) Split component-wise:
#       - normal CC: assign whole CC by centroid z vs cc_thr
#       - suspicious CC that spans both sides:
#           (a) DT-core split (NON-destructive): find r where dt>=r breaks the bridge -> use the two cores as seeds
#           (b) band-cut (non-destructive assignment for removed band) for very large blobs
#           (c) erosion split fallback (non-destructive reconstruction)
#  5) Optional refine near interface: RELABEL only (do NOT delete voxels; interface_gap default 0)
#
# Usage:
#   python split_jaws_teeth_only.py --patient_dir "D:\...\data\52730449" --min_global 13000 --swap
# Recommended (avoid damage):
#   --interface_gap 0

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
    if min_voxels <= 0:
        return bin_mask.astype(bool)
    structure = ndimage.generate_binary_structure(3, 2)  # 26-connected
    labeled, num = ndimage.label(bin_mask, structure=structure)
    if num == 0:
        return bin_mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[1:] = counts[1:] >= min_voxels
    return keep[labeled]


def keep_top_k_components_3d(bin_mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return bin_mask.astype(bool)
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(bin_mask, structure=structure)
    if num == 0:
        return bin_mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    top = np.argsort(counts)[::-1][:k]
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


# ----------------------------
# Plane selection
# ----------------------------
def axis0_profile(teeth: np.ndarray) -> np.ndarray:
    return teeth.sum(axis=(1, 2)).astype(np.float32)


def find_plane_axis0_by_profile(teeth: np.ndarray, smooth_win: int = 21) -> int:
    prof_s = smooth_1d(axis0_profile(teeth), smooth_win)
    d = len(prof_s)
    mid = d // 2
    p1 = int(np.argmax(prof_s[:mid]))
    p2 = int(np.argmax(prof_s[mid:]) + mid)
    if p1 > p2:
        p1, p2 = p2, p1
    valley = int(np.argmin(prof_s[p1:p2 + 1]) + p1)
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
    below = int(teeth[:plane, :, :].sum())
    above = int(teeth[plane:, :, :].sum())
    total = below + above
    min_frac = (min(below, above) / total) if total > 0 else 0.0
    return float(min_frac), below, above, total


def refine_plane_to_local_valley(teeth: np.ndarray, plane0: int, smooth_win: int = 21, radius: int = 30) -> int:
    prof_s = smooth_1d(axis0_profile(teeth), smooth_win)
    d = len(prof_s)
    lo = max(0, int(plane0) - int(radius))
    hi = min(d, int(plane0) + int(radius) + 1)
    if hi <= lo + 1:
        return int(np.clip(plane0, 0, d - 1))
    local = int(np.argmin(prof_s[lo:hi]) + lo)
    return local


def choose_plane_axis0(teeth: np.ndarray, smooth_win: int = 21, refine_radius: int = 30) -> int:
    if smooth_win is None or smooth_win <= 0:
        smooth_win = 21

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
def try_split_blob_by_erosion(sub: np.ndarray, min_sub_vox: int = 2000, erosion_max: int = 10, structure=None):
    """Erode until it splits -> use two largest as seeds -> reconstruct by nearest-seed (NON-destructive)."""
    if structure is None:
        structure = ndimage.generate_binary_structure(3, 2)
    min_sub_vox = int(max(0, min_sub_vox))
    erosion_max = int(max(0, erosion_max))
    if erosion_max <= 0:
        return None, None, 0

    sub = sub.astype(bool)
    total = int(sub.sum())
    if total == 0:
        return None, None, 0

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
    seed_margin: int = 10,
    r_max: int = 16,
    min_seed_vox: int = 1200,
    min_part_vox: int = 1500,
    structure=None,
):
    """
    NON-destructive: use distance-to-background (EDT) to find "thick core" that breaks the thin bridge.

    Steps:
      1) Compute dt = EDT(sub)
      2) For r = 2..r_max: core = (dt >= r)
         As r increases, thin connections disappear earlier than thick tooth bodies.
      3) Find best r where core has TWO different CCs connected to upper_seed and lower_seed respectively.
      4) Use those two core CCs as seeds, reconstruct full partition of ORIGINAL sub by nearest-seed.

    Returns: (part_upper, part_lower, best_r) in LOCAL coords, or (None,None,0).
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

    zz = np.arange(Z)[:, None, None]

    # Convention: smaller z = upper, larger z = lower
    upper_seed = sub & (zz <= plane_local - seed_margin)
    lower_seed = sub & (zz >= plane_local + seed_margin)

    if int(upper_seed.sum()) < int(min_seed_vox) or int(lower_seed.sum()) < int(min_seed_vox):
        return None, None, 0

    dt = ndimage.distance_transform_edt(sub)
    dt_max = float(dt.max())
    if dt_max < 2.0:
        return None, None, 0

    best = None  # (r, id_u, id_l, lab_core, core_sizes)
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

        # find dominant core id that overlaps each seed
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

        # this r works; keep the largest r (most aggressive core shrink) that still splits
        best = (r, id_u, id_l, labc)

    if best is None:
        return None, None, 0

    r_best, id_u, id_l, labc = best
    seed_u = (labc == id_u)
    seed_l = (labc == id_l)

    # Reconstruct full partition of ORIGINAL sub by nearest seed (NON-destructive)
    d_u = ndimage.distance_transform_edt(~seed_u)
    d_l = ndimage.distance_transform_edt(~seed_l)

    part_u = sub & (d_u <= d_l)
    part_l = sub & (d_l < d_u)

    if int(part_u.sum()) < min_part_vox or int(part_l.sum()) < min_part_vox:
        return None, None, 0

    return part_u, part_l, int(r_best)


# ----------------------------
# Component-wise split
# ----------------------------
def split_jaws_axis0_componentwise(
    teeth: np.ndarray,
    plane: int,
    cc_thr: float,
    cut_half_thickness: int = 3,
    span_min_frac: float = 0.10,
    try_cut_min_vox: int = 25000,
    min_sub_vox: int = 3000,
    big_comp_vox: int = 80000,
    erosion_max: int = 10,
    erosion_min_sub_vox: int = 2000,
    # DT-core split params
    seed_margin: int = 10,
    refine_band: int = 18,          # used only to decide "crosses_by_margin" robustness (via seed_margin)
    dtcore_try_min_vox: int = 6000, # try dt-core split for CCs above this size
):
    teeth = teeth.astype(bool)
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, num = ndimage.label(teeth, structure=structure)
    if num == 0:
        return np.zeros_like(teeth, bool), np.zeros_like(teeth, bool)

    objs = ndimage.find_objects(labeled)
    upper = np.zeros_like(teeth, dtype=bool)
    lower = np.zeros_like(teeth, dtype=bool)

    plane = int(np.clip(plane, 0, teeth.shape[0] - 1))
    cc_thr = float(cc_thr)

    cut_half_thickness = int(max(0, cut_half_thickness))
    thr_plane = int(np.clip(int(round(cc_thr)), 0, teeth.shape[0] - 1))
    try_cut_min_vox = int(min(try_cut_min_vox, big_comp_vox))

    seed_margin = int(max(0, seed_margin))
    dtcore_try_min_vox = int(max(0, dtcore_try_min_vox))

    for lab in range(1, num + 1):
        slc = objs[lab - 1]
        if slc is None:
            continue

        sub = (labeled[slc] == lab)
        if not np.any(sub):
            continue

        total = int(sub.sum())
        z0, z1 = slc[0].start, slc[0].stop

        # quick z-range for robust "crosses_by_margin"
        zz_any = np.any(sub, axis=(1, 2))
        if np.any(zz_any):
            zmin_local = int(np.argmax(zz_any))
            zmax_local = int(len(zz_any) - 1 - np.argmax(zz_any[::-1]))
            zmin_g = z0 + zmin_local
            zmax_g = z0 + zmax_local
        else:
            zmin_g = z0
            zmax_g = z1 - 1

        crosses_by_margin = (zmin_g <= (cc_thr - seed_margin)) and (zmax_g >= (cc_thr + seed_margin))
        plane_local = thr_plane - z0

        # (A) DT-core split (NON-destructive) – best for thin/slanted bridges + tiny thin slice issues
        if crosses_by_margin and total >= dtcore_try_min_vox and erosion_max > 0:
            pu, pl, r_used = try_split_blob_by_dt_core(
                sub,
                plane_local=plane_local,
                seed_margin=seed_margin,
                r_max=erosion_max,
                min_seed_vox=erosion_min_sub_vox,
                min_part_vox=max(1500, min_sub_vox),
                structure=structure,
            )
            if pu is not None:
                # assign by centroid vs cc_thr (safety)
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

        # (B) Classic spans_both criterion for big blobs
        spans_both = False
        if thr_plane > z0 and thr_plane < z1:
            lp = thr_plane - z0
            below = int(sub[:lp, :, :].sum())
            above = total - below
            spans_both = (min(below, above) / max(total, 1)) > float(span_min_frac)

        # (C) band-cut for very large spanning blob (still non-destructive overall: only removes band for splitting then reassign)
        if spans_both and total >= try_cut_min_vox and cut_half_thickness > 0:
            sub_cut = sub.copy()
            cut_plane_g = int(np.clip(thr_plane, z0, z1 - 1))
            band_lo_g = cut_plane_g - cut_half_thickness
            band_hi_g = cut_plane_g + cut_half_thickness + 1
            lo = max(0, band_lo_g - z0)
            hi = min(z1 - z0, band_hi_g - z0)
            if hi > lo:
                sub_cut[lo:hi, :, :] = False

            sub_labeled, sub_num = ndimage.label(sub_cut, structure=structure)
            if sub_num >= 2:
                sub_sizes = np.bincount(sub_labeled.ravel())
                keep_ids = [sid for sid in range(1, sub_num + 1) if sub_sizes[sid] >= max(1500, min_sub_vox)]

                if len(keep_ids) >= 2:
                    parts = []
                    for sid in keep_ids:
                        m = (sub_labeled == sid)
                        if not np.any(m):
                            continue
                        cz_local = ndimage.center_of_mass(m)[0]
                        cz = float(cz_local + z0)
                        parts.append((sid, int(sub_sizes[sid]), cz))

                    if len(parts) >= 2:
                        has_low = any(p[2] < cc_thr for p in parts)
                        has_high = any(p[2] >= cc_thr for p in parts)
                        cz_min = min(p[2] for p in parts)
                        cz_max = max(p[2] for p in parts)
                        if has_low and has_high and (cz_max - cz_min) >= 6.0:
                            for sid, _, cz in parts:
                                m = (sub_labeled == sid)
                                if cz < cc_thr:
                                    upper[slc] |= m
                                else:
                                    lower[slc] |= m

                            # reassign the removed band voxels by z (no truncation elsewhere)
                            band = sub & (~sub_cut)
                            if np.any(band):
                                z_idx_g = np.arange(z0, z1)[:, None, None].astype(np.float32)
                                upper[slc] |= band & (z_idx_g < cc_thr)
                                lower[slc] |= band & (z_idx_g >= cc_thr)
                            continue

        # (D) erosion fallback for big spanning blobs
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
                if ((cz0 < cc_thr) != (cz1 < cc_thr)) and (abs(cz0 - cz1) >= 6.0):
                    if cz0 < cc_thr:
                        upper[slc] |= p0
                        lower[slc] |= p1
                    else:
                        upper[slc] |= p1
                        lower[slc] |= p0
                    continue

        # Normal CC: assign WHOLE CC by centroid
        cz = float(ndimage.center_of_mass(sub)[0] + z0)
        if cz < cc_thr:
            upper[slc] |= sub
        else:
            lower[slc] |= sub

    return upper, lower


# ----------------------------
# Refinement near interface (NON-destructive recommended)
# ----------------------------
def refine_interface_by_distance(
    teeth: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    cc_thr: float,
    refine_band: int = 12,
    seed_margin: int = 6,
    interface_gap: float = 0.0,  # RECOMMENDED 0 (do not delete voxels)
    roi_pad: int = 6,
):
    """
    RELABEL only in band around cc_thr using distance to stable seeds.
    If interface_gap>0, it will delete voxels at interface (can damage surfaces) – not recommended.
    """
    teeth = teeth.astype(bool)
    upper = upper.astype(bool)
    lower = lower.astype(bool)

    Z = teeth.shape[0]
    thr = float(cc_thr)

    refine_band = int(max(0, refine_band))
    seed_margin = int(max(0, seed_margin))
    roi_pad = int(max(0, roi_pad))

    if refine_band <= 0:
        return upper, lower

    half = refine_band + seed_margin + roi_pad
    z_lo = int(max(0, np.floor(thr - half)))
    z_hi = int(min(Z, np.ceil(thr + half) + 1))

    teeth_r = teeth[z_lo:z_hi]
    upper_r = upper[z_lo:z_hi]
    lower_r = lower[z_lo:z_hi]

    z_g = np.arange(z_lo, z_hi, dtype=np.float32)[:, None, None]
    band = teeth_r & (np.abs(z_g - thr) <= float(refine_band))

    # Seeds: upper = smaller z, lower = larger z
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

    # NOT recommended; keep 0 to avoid damage
    if interface_gap and interface_gap > 0:
        interface = band & (np.abs(du - dl) <= float(interface_gap))
        new_upper_r[interface] = False
        new_lower_r[interface] = False

    upper2 = upper.copy()
    lower2 = lower.copy()
    upper2[z_lo:z_hi] = new_upper_r
    lower2[z_lo:z_hi] = new_lower_r

    upper2 &= teeth
    lower2 &= teeth

    overlap = upper2 & lower2
    if np.any(overlap):
        oz, oy, ox = np.nonzero(overlap)
        pick_upper = (oz.astype(np.float32) < float(thr))
        upper2[oz, oy, ox] = pick_upper
        lower2[oz, oy, ox] = ~pick_upper

    return upper2, lower2


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_dir", type=str, default=None,
                    help=r"病人資料夾（內含 mask_binary.npy），例如 D:\\...\\data\\57969132")
    ap.add_argument("--mask", type=str, default=None,
                    help="mask_binary.npy 完整路徑（提供則忽略 --patient_dir）")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="輸出資料夾（預設同 patient_dir）")

    # cleaning
    ap.add_argument("--min_global", type=int, default=3000)
    ap.add_argument("--keep_top_k_global", type=int, default=0)
    ap.add_argument("--min_per_jaw", type=int, default=1000,
                    help="分顎後清理小區塊（不要太大，避免根尖被當小塊丟掉）")
    ap.add_argument("--keep_top_k_per_jaw", type=int, default=0)

    # plane
    ap.add_argument("--plane", type=int, default=-1, help="手動指定 plane (axis0 index)，-1 表示自動")
    ap.add_argument("--smooth", type=int, default=21, help="profile 平滑窗（11~31）")
    ap.add_argument("--refine_radius", type=int, default=30, help="plane 局部 valley refinement 範圍")

    # split behavior
    ap.add_argument("--cut_half_thickness", type=int, default=4,
                    help="只對『大且跨 plane 的 blob』在 plane 附近挖空帶半厚度（2~6）")
    ap.add_argument("--ambig_frac", type=float, default=0.15,
                    help="判定『大 blob 跨 plane』的比例門檻（min(below,above)/total）。0.05~0.20")
    ap.add_argument("--big_comp_vox", type=int, default=80000,
                    help="只有超過此體積且跨 plane 的 component 才會做 band-cut")
    ap.add_argument("--try_cut_min_vox", type=int, default=25000,
                    help="嘗試 band-cut 的最小體積門檻")
    ap.add_argument("--min_sub_vox", type=int, default=3000,
                    help="子 component 最小體積（避免碎片）。1000~5000")

    # interface refinement
    ap.add_argument("--refine_band", type=int, default=14,
                    help="cc_thr 附近 band 內重標籤。建議 10~20")
    ap.add_argument("--seed_margin", type=int, default=8,
                    help="穩定種子距離。建議 6~12")
    ap.add_argument("--interface_gap", type=float, default=0.0,
                    help="不建議 >0（會挖洞破壞牙面）。建議 0。")

    # fallback
    ap.add_argument("--erosion_max", type=int, default=10,
                    help="erosion fallback 最大迭代數（6~16）")
    ap.add_argument("--erosion_min_sub_vox", type=int, default=2000,
                    help="erosion 分裂 seeds 最小體積（1500~4000）")

    ap.add_argument("--swap", action="store_true", help="交換 upper/lower（如果你視覺上顛倒）")

    args = ap.parse_args()

    if args.mask is None and args.patient_dir is None:
        raise ValueError("請提供 --patient_dir 或 --mask")

    if args.mask is not None:
        mask_path = args.mask
        base_dir = args.out_dir if args.out_dir else os.path.dirname(mask_path)
        out_dir = args.out_dir if args.out_dir else base_dir
    else:
        base_dir = args.patient_dir
        mask_path = os.path.join(base_dir, "mask_binary.npy")
        out_dir = args.out_dir if args.out_dir else base_dir

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
    print("after global clean voxels:", int(teeth.sum()))

    # choose plane (axis=0 fixed)
    if args.plane >= 0:
        plane = int(np.clip(args.plane, 0, teeth.shape[0] - 1))
        print("[plane] manual:", plane)
    else:
        plane = choose_plane_axis0(teeth, smooth_win=args.smooth, refine_radius=args.refine_radius)
        print("[plane] auto:", plane)

    mf, below, above, total = eval_plane(teeth, plane)
    print(f"[plane] voxels below={below}, above={above}, total={total}, min_side_frac={mf:.3f}")

    # threshold by CC centroids
    cc_thr, cc_c0, cc_c1, cc_num = cc_kmeans_threshold_axis0(teeth)
    print(f"[cc_kmeans] num_cc={cc_num}, centers=({cc_c0:.2f},{cc_c1:.2f}), thr={cc_thr:.2f}")

    # split
    upper, lower = split_jaws_axis0_componentwise(
        teeth,
        plane=plane,
        cc_thr=cc_thr,
        cut_half_thickness=args.cut_half_thickness,
        span_min_frac=args.ambig_frac,
        try_cut_min_vox=args.try_cut_min_vox,
        min_sub_vox=args.min_sub_vox,
        big_comp_vox=args.big_comp_vox,
        erosion_max=args.erosion_max,
        erosion_min_sub_vox=args.erosion_min_sub_vox,
        seed_margin=args.seed_margin,
        refine_band=args.refine_band,
        dtcore_try_min_vox=max(6000, args.min_sub_vox),
    )

    # refine (NON-destructive recommended: interface_gap=0)
    if args.refine_band and args.refine_band > 0:
        upper, lower = refine_interface_by_distance(
            teeth, upper, lower,
            cc_thr=cc_thr,
            refine_band=args.refine_band,
            seed_margin=args.seed_margin,
            interface_gap=args.interface_gap,
            roi_pad=6,
        )

    # per-jaw clean
    upper = remove_small_components_3d(upper, args.min_per_jaw)
    lower = remove_small_components_3d(lower, args.min_per_jaw)
    upper = keep_top_k_components_3d(upper, args.keep_top_k_per_jaw)
    lower = keep_top_k_components_3d(lower, args.keep_top_k_per_jaw)

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
