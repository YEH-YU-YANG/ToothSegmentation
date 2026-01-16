# split_jaws/refine.py
import numpy as np
from scipy import ndimage as ndi
from .cc_utils import get_structure
from .watershed_cost import make_cost_from_edt


def _relax_seed_dt(edt_vals: np.ndarray, seed_dt_min: float, percentile: float) -> float:
    if edt_vals.size == 0:
        return seed_dt_min
    p = float(np.percentile(edt_vals, percentile))
    return max(seed_dt_min, p)


def band_refine_watershed(
    teeth_mask: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    thr_z: float,
    refine_band: int,
    seed_margin: int,
    seed_dt_min: float,
    seed_dt_percentile: float,
    cost_gamma: float,
    connectivity: int = 26,
    min_seed_vox: int = 200,
    debug: bool = False,
):
    """
    Only relabel voxels inside |z-thr|<=refine_band using thickness-weighted watershed.
    Seeds come from confident regions outside the band (and thick-core).
    ROI is computed from thr/refine_band/seed_margin (FIXED) to ensure seeds exist in ROI.
    """
    if refine_band <= 0:
        if debug:
            print(f"[refine] skip: refine_band={refine_band}", flush=True)
        return upper, lower

    teeth = (teeth_mask > 0)
    Z = teeth.shape[0]
    z = np.arange(Z, dtype=np.float32)
    band = (np.abs(z - float(thr_z)) <= float(refine_band))[:, None, None] & teeth

    if band.sum() == 0:
        if debug:
            print("[refine] skip: band is empty", flush=True)
        return upper, lower

    # FIXED ROI: include confident seed regions outside band
    pad = refine_band + seed_margin + 2
    z0 = max(0, int(np.floor(thr_z - pad)))
    z1 = min(Z, int(np.ceil(thr_z + pad)) + 1)
    roi = (slice(z0, z1), slice(None), slice(None))

    if debug:
        print(
            f"[refine] band_vox={int(band.sum())} roi_z=[{z0},{z1}) thr={thr_z:.2f} "
            f"refine_band={refine_band} seed_margin={seed_margin} min_seed_vox={min_seed_vox}",
            flush=True,
        )

    teeth_roi = teeth[roi].astype(np.uint8)
    if teeth_roi.sum() == 0:
        if debug:
            print("[refine] skip: teeth_roi is empty", flush=True)
        return upper, lower

    edt = ndi.distance_transform_edt(teeth_roi)
    edt_vals = edt[teeth_roi > 0]
    seed_dt = _relax_seed_dt(edt_vals, seed_dt_min, seed_dt_percentile)

    z_roi = np.arange(teeth_roi.shape[0], dtype=np.float32) + float(z0)
    z_grid = z_roi[:, None, None]

    # confident seeds well outside the band
    upper_conf = (upper[roi] > 0) & (z_grid >= (thr_z + refine_band + seed_margin))
    lower_conf = (lower[roi] > 0) & (z_grid <= (thr_z - refine_band - seed_margin))

    upper_seed = upper_conf & (edt >= seed_dt)
    lower_seed = lower_conf & (edt >= seed_dt)

    if debug:
        print(
            f"[refine] seed_dt={seed_dt:.2f} (min={seed_dt_min}, pct={seed_dt_percentile}) "
            f"upper_conf={int(upper_conf.sum())} lower_conf={int(lower_conf.sum())} "
            f"upper_seed={int(upper_seed.sum())} lower_seed={int(lower_seed.sum())}",
            flush=True,
        )

    # relax if too few seeds
    if upper_seed.sum() < min_seed_vox or lower_seed.sum() < min_seed_vox:
        for pct in [max(50.0, seed_dt_percentile - 20.0), 40.0, 30.0, 20.0]:
            seed_dt2 = _relax_seed_dt(edt_vals, max(0.5, 0.5 * seed_dt_min), pct)
            upper_seed = upper_conf & (edt >= seed_dt2)
            lower_seed = lower_conf & (edt >= seed_dt2)
            if debug:
                print(
                    f"[refine] relax pct={pct:.0f} seed_dt={seed_dt2:.2f} "
                    f"upper_seed={int(upper_seed.sum())} lower_seed={int(lower_seed.sum())}",
                    flush=True,
                )
            if upper_seed.sum() >= min_seed_vox and lower_seed.sum() >= min_seed_vox:
                seed_dt = seed_dt2
                break

    if upper_seed.sum() < min_seed_vox or lower_seed.sum() < min_seed_vox:
        if debug:
            print(
                f"[refine] skip: not enough seeds (upper={int(upper_seed.sum())}, lower={int(lower_seed.sum())}, need>={min_seed_vox})",
                flush=True,
            )
        return upper, lower

    markers = np.zeros(teeth_roi.shape, dtype=np.int32)
    markers[lower_seed] = 1
    markers[upper_seed] = 2

    cost = make_cost_from_edt(edt, gamma=cost_gamma)
    cost2 = cost.copy()
    cost2[teeth_roi == 0] = np.uint16(65535)

    struct = get_structure(connectivity)
    lab = ndi.watershed_ift(cost2, markers=markers, structure=struct).astype(np.uint8)

    # apply only to band voxels
    band_roi = band[roi] & (teeth_roi > 0)
    if not band_roi.any():
        if debug:
            print("[refine] skip: band_roi is empty", flush=True)
        return upper, lower

    if debug:
        print(f"[refine] apply: band_roi_vox={int(band_roi.sum())}", flush=True)

    upper2 = upper.copy()
    lower2 = lower.copy()

    # clear old labels in band, then assign from watershed
    upper2[roi][band_roi] = 0
    lower2[roi][band_roi] = 0

    upper2[roi][band_roi & (lab == 2)] = 1
    lower2[roi][band_roi & (lab == 1)] = 1

    return upper2.astype(np.uint8), lower2.astype(np.uint8)
