# split_jaws/watershed_cost.py
import numpy as np

def make_cost_from_edt(edt: np.ndarray, gamma: float = 2.0, eps: float = 1e-3) -> np.ndarray:
    """
    EDT thick => low cost; EDT thin => high cost
    cost ~ (1/(edt+eps))^gamma, scaled to uint16 for watershed_ift
    """
    e = edt.astype(np.float32, copy=False)
    inv = 1.0 / (e + eps)
    cost = np.power(inv, gamma)

    finite = np.isfinite(cost)
    cmax = float(cost[finite].max()) if finite.any() else 1.0
    cmax = max(cmax, 1e-9)

    cn = np.clip(cost / cmax, 0.0, 1.0)
    return (cn * 65535.0).astype(np.uint16)
