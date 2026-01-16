# split_jaws/thr_estimation.py
import numpy as np
from scipy import ndimage as ndi
from .cc_utils import get_structure

def estimate_thr_from_profile(mask: np.ndarray) -> float:
    prof = (mask > 0).sum(axis=(1, 2)).astype(np.float64)
    if prof.size < 10:
        return float(prof.size // 2)

    prof_s = ndi.gaussian_filter1d(prof, sigma=2.0)
    mid = prof_s.size // 2

    p1 = int(np.argmax(prof_s[:mid])) if mid > 0 else 0
    p2 = int(np.argmax(prof_s[mid:]) + mid) if mid < prof_s.size else int(np.argmax(prof_s))

    if p2 <= p1:
        return float(mid)

    valley = int(np.argmin(prof_s[p1:p2 + 1]) + p1)
    return float(valley)

def _weighted_kmeans_1d(values: np.ndarray, weights: np.ndarray, iters: int = 30):
    v = values.astype(np.float64)
    w = weights.astype(np.float64)
    if v.size < 2:
        c = float(v[0]) if v.size == 1 else 0.0
        return c, c

    c1 = float(np.percentile(v, 25))
    c2 = float(np.percentile(v, 75))
    if abs(c2 - c1) < 1e-3:
        c1, c2 = float(v.min()), float(v.max())

    for _ in range(iters):
        d1 = np.abs(v - c1)
        d2 = np.abs(v - c2)
        a = d1 <= d2
        if a.all() or (~a).all():
            break
        w1 = w[a].sum()
        w2 = w[~a].sum()
        nc1 = (v[a] * w[a]).sum() / max(w1, 1e-9)
        nc2 = (v[~a] * w[~a]).sum() / max(w2, 1e-9)
        if abs(nc1 - c1) < 1e-4 and abs(nc2 - c2) < 1e-4:
            c1, c2 = nc1, nc2
            break
        c1, c2 = nc1, nc2

    return (c1, c2) if c1 <= c2 else (c2, c1)

def estimate_thr_from_cc(mask: np.ndarray, connectivity: int = 26, min_cc_for_stats: int = 500):
    """
    Returns thr_z, info_dict
    """
    struct = get_structure(connectivity)
    lbl, n = ndi.label(mask > 0, structure=struct)
    if n == 0:
        return 0.0, {"n_cc": 0, "mode": "empty"}

    objs = ndi.find_objects(lbl)
    centroids = []
    weights = []

    for i, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        sub = (lbl[slc] == i)
        cnt = int(sub.sum())
        if cnt < min_cc_for_stats:
            continue
        zz = np.nonzero(sub)[0].astype(np.float64) + slc[0].start
        cz = float(zz.mean()) if zz.size else float((slc[0].start + slc[0].stop - 1) / 2.0)
        centroids.append(cz)
        weights.append(cnt)

    if len(centroids) < 2:
        thr = estimate_thr_from_profile(mask)
        return thr, {"n_cc": n, "mode": "profile_fallback", "centroids_used": len(centroids)}

    centroids = np.asarray(centroids, np.float64)
    weights = np.asarray(weights, np.float64)
    c_low, c_high = _weighted_kmeans_1d(centroids, weights)
    thr_k = 0.5 * (c_low + c_high)

    if abs(c_high - c_low) < 3.0:
        thr = estimate_thr_from_profile(mask)
        return thr, {"n_cc": n, "mode": "profile_override", "centers": (c_low, c_high), "thr_kmeans": thr_k}

    return float(thr_k), {"n_cc": n, "mode": "kmeans", "centers": (c_low, c_high), "thr": float(thr_k)}
