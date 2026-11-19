"""Export a scan folder as a COLMAP text model with known poses.

This feeds the COLMAP + 3DGS baseline the exact same images and poses as the
Go reconstruction:

    python -m plant_capture.tools.export_colmap scan_dir out_dir/sparse/0
    colmap point_triangulator --database_path db.db --image_path scan_dir/frames \
        --input_path out_dir/sparse/0 --output_path out_dir/sparse/tri

(COLMAP needs features in the database first: run feature_extractor with
--ImageReader.camera_model matching cameras.txt and exhaustive_matcher.)
For the "COLMAP poses" baseline, ignore this exporter and run normal SfM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ..core.dataset_reader import Dataset, load_dataset
from ..core.geometry import matrix_to_quat, rigid_inverse
from ..core.model import FrameRecord, Intrinsics

CAMERA_ID = 1


def camera_line(k: Intrinsics) -> str:
    """cameras.txt entry. OPENCV has k1 k2 p1 p2; FULL_OPENCV is needed if k3 ≠ 0."""
    k1, k2, p1, p2, k3 = k.distortion
    if k3 == 0:
        params = [k.fx, k.fy, k.cx, k.cy, k1, k2, p1, p2]
        model = "OPENCV"
    else:
        params = [k.fx, k.fy, k.cx, k.cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0]
        model = "FULL_OPENCV"
    return " ".join([str(CAMERA_ID), model, str(k.width), str(k.height), *(repr(float(p)) for p in params)])


def image_lines(image_id: int, frame: FrameRecord) -> tuple[str, str]:
    """images.txt entry: world-to-camera rotation as (qw qx qy qz) and
    translation, then an empty line of 2D points."""
    T_cam_world = rigid_inverse(frame.T_world_cam)
    qx, qy, qz, qw = matrix_to_quat(T_cam_world[:3, :3])
    tx, ty, tz = T_cam_world[:3, 3]
    name = Path(frame.image).name
    values = " ".join(repr(float(v)) for v in (qw, qx, qy, qz, tx, ty, tz))
    return f"{image_id} {values} {CAMERA_ID} {name}", ""


def write_lines(path: Path, header: str, lines: Iterable[str]) -> None:
    path.write_text(header + "".join(f"{line}\n" for line in lines), encoding="utf-8")


def export(dataset: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(out_dir / "cameras.txt", "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n",
                [camera_line(dataset.intrinsics)])
    write_lines(out_dir / "images.txt",
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X Y POINT3D_ID)\n",
                [line for i, f in enumerate(dataset.frames, start=1) for line in image_lines(i, f)])
    write_lines(out_dir / "points3D.txt", "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n", [])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    ds = load_dataset(args.scan_dir)
    export(ds, args.out_dir)
    print(f"exported {len(ds.frames)} images to {args.out_dir}")


if __name__ == "__main__":
    main()
