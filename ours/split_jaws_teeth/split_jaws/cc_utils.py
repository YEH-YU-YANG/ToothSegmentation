# split_jaws/cc_utils.py
import numpy as np
from scipy import ndimage as ndi

def get_structure(connectivity: int = 26):
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    # 26-connectivity
    return ndi.generate_binary_structure(3, 2)

def remove_small_cc(mask: np.ndarray, min_vox: int, connectivity: int = 26) -> np.ndarray:
    if min_vox <= 0:
        return mask.astype(np.uint8, copy=False)

    struct = get_structure(connectivity)
    lbl, n = ndi.label(mask > 0, structure=struct)
    if n == 0:
        return (mask > 0).astype(np.uint8)

    counts = np.bincount(lbl.ravel())
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    keep[counts < min_vox] = False

    return keep[lbl].astype(np.uint8)

def bbox_pad(slc, shape, pad: int):
    z0 = max(0, slc[0].start - pad); z1 = min(shape[0], slc[0].stop + pad)
    y0 = max(0, slc[1].start - pad); y1 = min(shape[1], slc[1].stop + pad)
    x0 = max(0, slc[2].start - pad); x1 = min(shape[2], slc[2].stop + pad)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))

def cc_stats_from_label(lbl: np.ndarray, cc_id: int):
    pos = np.nonzero(lbl == cc_id)
    if len(pos[0]) == 0:
        return None
    z = pos[0]
    zmin = int(z.min())
    zmax = int(z.max())
    cz = float(z.mean())
    vox = int(z.size)
    return zmin, zmax, cz, vox
