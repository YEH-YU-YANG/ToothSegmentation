import os
import re
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


def natural_key_from_relative(path: Path, root_dir: Path):
    rel = str(path.relative_to(root_dir))
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.findall(r"\d+|\D+", rel))


def contains_ios(p: Path) -> bool:
    s = str(p).lower()
    return "ios" in s


def is_cbct_series_dir(series_dir: Path, dicom_root: Path) -> bool:
    try:
        rel_parts = series_dir.relative_to(dicom_root).parts
    except ValueError:
        rel_parts = series_dir.parts

    parts_lower = [x.lower() for x in rel_parts]

    # Must be under a CBCT folder, and not under anything that looks like IOS
    if "cbct" not in parts_lower:
        return False
    if any("ios" in x for x in parts_lower):
        return False
    return True


def output_rel_for_cbct(series_dir: Path, dicom_root: Path) -> Path | None:
    """Return output relative path like <patient_id>/dcm_to_png/<date>.

    Accepts both:
      - <id>/CBCT/<date>
      - <id>/CBCT/<date>/DCM (or /DICOM)
    """
    try:
        rel = series_dir.relative_to(dicom_root)
    except ValueError:
        return None

    parts = rel.parts
    parts_lower = [x.lower() for x in parts]

    if not parts:
        return None

    if any("ios" in x for x in parts_lower):
        return None

    if "cbct" not in parts_lower:
        return None

    patient_id = parts[0]
    cbct_idx = parts_lower.index("cbct")

    date_part = None
    if cbct_idx + 1 < len(parts):
        candidate = parts[cbct_idx + 1]
        if candidate.lower() not in {"dcm", "dicom"}:
            date_part = candidate

    # If the series dir ends with DCM/DICOM, prefer the folder right after CBCT as date
    if parts_lower[-1] in {"dcm", "dicom"}:
        if cbct_idx + 1 < len(parts) - 1:
            date_part = parts[cbct_idx + 1]

    out_rel = Path(patient_id) / "dcm_to_png"
    if date_part is not None:
        out_rel = out_rel / date_part
    return out_rel


def collect_series_dirs(dicom_root: Path, cbct_only: bool = True):
    dicom_root = Path(dicom_root)
    dcm_files = list(dicom_root.rglob("*.dcm")) + list(dicom_root.rglob("*.DCM"))

    series_dirs: set[Path] = set()
    for f in dcm_files:
        parent = f.parent
        if cbct_only and not is_cbct_series_dir(parent, dicom_root):
            continue
        series_dirs.add(parent)

    return sorted(series_dirs, key=lambda p: natural_key_from_relative(p, dicom_root))


def load_volume(volume_dir: Path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(volume_dir))
    if len(series_ids) == 0:
        raise ValueError(f'No DICOM series found in "{volume_dir}"')

    series_id = series_ids[0]
    filenames = reader.GetGDCMSeriesFileNames(str(volume_dir), series_id)
    filenames = [fn for fn in filenames if fn.lower().endswith(".dcm")]
    if len(filenames) == 0:
        raise ValueError(f'No .dcm files selected in "{volume_dir}"')

    reader.SetFileNames(filenames)
    return reader.Execute()


def preprocess_volume(volume, min_value: int = -1000, max_value: int = 4500):
    volume = sitk.IntensityWindowing(volume, min_value, max_value)
    volume = sitk.Cast(volume, sitk.sitkUInt8)
    return volume


def resample_volume(volume, new_spacing=(0.25, 0.25, 0.25)):
    size = np.array(volume.GetSize(), dtype=np.float64)
    spacing = np.array(volume.GetSpacing(), dtype=np.float64)
    new_spacing = np.array(list(new_spacing), dtype=np.float64)

    new_size = np.round(size * spacing / new_spacing).astype(int).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(volume)
    resample.SetOutputSpacing([float(x) for x in new_spacing])
    resample.SetSize([int(x) for x in new_size])
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(volume)


def save_volume(volume, output_dir: Path, *, flip_lr: bool = False):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    arr = sitk.GetArrayFromImage(volume)  # [z, y, x]
    for idx, img in enumerate(arr, start=0):
        # If you want the PNG to visually match the typical STL occlusal-view
        # (what you see in most mesh viewers), you usually want the
        # *neurological* convention: image-left == patient-left.
        # Many DICOM viewers use the *radiological* convention
        # (image-left == patient-right). So we optionally flip L/R here.
        if flip_lr:
            img = img[:, ::-1]

        out_path = output_dir / f"{idx:04d}.png"

        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def main():
    from argparse import ArgumentParser

    # Prefer the same UX as your reference script, but also support --dicom/--output.
    try:
        from console import Table, track  # type: ignore
    except Exception:
        from contextlib import contextmanager
        from tqdm import tqdm

        class _FallbackTable:
            def __init__(self, headers):
                self.headers = headers
                self.rows = []

            def add_row(self, row):
                self.rows.append(row)

            def __enter__(self):
                print("\t".join(map(str, self.headers)))
                return self

            def __exit__(self, exc_type, exc, tb):
                for r in self.rows:
                    print("\t".join(map(str, r)))
                return False

        @contextmanager
        def Table(headers):
            t = _FallbackTable(headers)
            try:
                yield t
            finally:
                t.__exit__(None, None, None)

        def track(iterable):
            return tqdm(iterable)

    parser = ArgumentParser()

    # Positional (compatible with your reference code)
    parser.add_argument("dicom", nargs="?")
    parser.add_argument("output", nargs="?")

    # Optional flags (compatible with your current call)
    parser.add_argument("--dicom", dest="dicom_opt", type=str)
    parser.add_argument("--output", dest="output_opt", type=str)

    parser.add_argument("--min", type=int, default=-1000)
    parser.add_argument("--max", type=int, default=4500)
    parser.add_argument("--spacing", type=float, default=0.25)

    # Display convention for saved PNGs.
    # - stl  : image-left == patient-left (often matches STL occlusal-view screenshots)
    # - dicom: image-left == patient-right (radiological convention; common in many DICOM viewers)
    parser.add_argument(
        "--view",
        type=str,
        choices=["stl", "dicom"],
        default="stl",
        help="PNG display convention. 'stl' matches typical STL occlusal view (default).",
    )

    parser.add_argument(
        "--no_cbct_filter",
        action="store_true",
        help="Do not filter by CBCT (NOT recommended for your dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow output folder to exist (files may be overwritten).",
    )

    args = parser.parse_args()

    dicom_root = args.dicom_opt or args.dicom
    output_dir = args.output_opt or args.output
    if not dicom_root or not output_dir:
        parser.error("Please provide dicom root and output folder (positional or --dicom/--output).")

    dicom_root = Path(dicom_root)
    output_dir = Path(output_dir)

    # Unlike the original reference script, we allow the output root to already exist
    # (e.g. writing back into your dataset root). We guard overwriting at the per-case
    # folder level instead.
    output_dir.mkdir(parents=True, exist_ok=True)

    cbct_only = not args.no_cbct_filter
    series_dirs = collect_series_dirs(dicom_root, cbct_only=cbct_only)

    new_spacing = (args.spacing, args.spacing, args.spacing)

    with Table(["Output (relative)", "Source (relative)", "Status"]) as table:
        for series_dir in track(series_dirs):
            src_rel = series_dir.relative_to(dicom_root)

            if cbct_only:
                out_rel = output_rel_for_cbct(series_dir, dicom_root)
                if out_rel is None:
                    # Should not happen if cbct_only is True, but keep safe.
                    out_rel = src_rel
            else:
                out_rel = src_rel

            out_path = output_dir / out_rel

            # Do not overwrite an already converted case unless explicitly requested.
            if out_path.exists() and not args.overwrite:
                table.add_row([str(out_rel), str(src_rel), "SKIP: exists (use --overwrite)"])
                continue

            try:
                if out_path.exists() and not args.overwrite:
                    table.add_row([str(out_rel), str(src_rel), "SKIP: output exists (use --overwrite)"])
                    continue

                volume = load_volume(series_dir)
                volume = preprocess_volume(volume, args.min, args.max)
                volume = resample_volume(volume, new_spacing)
                # Match STL-style screenshots by default.
                save_volume(volume, out_path, flip_lr=(args.view.lower() == "stl"))
                table.add_row([str(out_rel), str(src_rel), "OK"])
            except Exception as e:
                # Skip problematic folders instead of crashing mid-way.
                table.add_row([str(out_rel), str(src_rel), f"SKIP: {type(e).__name__}"])


if __name__ == "__main__":
    main()
    
# python dcm2png.py --dicom "D:\YYYEH\TOOTHSEGMENTATION\data" --output "D:\YYYEH\TOOTHSEGMENTATION\data" --view stl --overwrite
# python dcm2png.py --dicom "D:\YYYEH\TOOTHSEGMENTATION\data\40657603" --output "D:\YYYEH\TOOTHSEGMENTATION\data\40657603" --view stl --overwrite