"""카메라 동작 확인용 스크립트.

실행:
    cd bin_picking_vision
    python scripts/capture_test.py --config config/config.yaml --out data/captures/test

확인 항목:
    1) 카메라 연결 OK
    2) 한 프레임 캡처 성공
    3) intensity image 저장 (PNG)
    4) PCD 저장 (PLY)
    5) 통계 출력 (유효 포인트 수, 거리 범위 등)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import open3d as o3d
import cv2
import numpy as np
import yaml

# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera import create_camera  # noqa: E402

def parse_args() -> argparse.Namespace: # 경로 설정
    p = argparse.ArgumentParser(description="LUCID Helios 동작 확인")
    p.add_argument(
        "--config", type=Path, default=ROOT / "config" / "config.yaml",
        help="config.yaml 경로",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "data" / "captures" / "test",
        help="출력 디렉토리",
    )
    p.add_argument(
        "--num-frames", type=int, default=1,
        help="캡처할 프레임 수 (warm-up 후)",
    )
    p.add_argument(
        "--warmup", type=int, default=3,
        help="버리는 첫 프레임 수 (ToF는 워밍업 권장)",
    )
    return p.parse_args()


def save_frame(frame, out_dir: Path, idx: int) -> None:
    """한 프레임을 PNG + PLY로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intensity image
    cv2.imwrite(str(out_dir / f"frame_{idx:04d}_intensity.png"), frame.intensity)

    # PCD를 PLY로 저장 (Open3D 사용)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        # 단위 변환: mm → m (Open3D 표준)
        pcd.points = o3d.utility.Vector3dVector(frame.points / 1000.0)
        o3d.io.write_point_cloud(
            str(out_dir / f"frame_{idx:04d}_pointcloud.ply"),
            pcd,
            write_ascii=False,
        )
    except ImportError:
        # Open3D 없으면 numpy로 저장
        np.save(out_dir / f"frame_{idx:04d}_points_mm.npy", frame.points)


def print_stats(frame, idx: int) -> None:
    """프레임 통계 출력."""
    valid_count = int(frame.valid_mask.sum())
    total = frame.height * frame.width
    pct = 100.0 * valid_count / total

    pts = frame.points
    if pts.size > 0:
        z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
        z_med = float(np.median(pts[:, 2]))
    else:
        z_min = z_max = z_med = float("nan")

    print(
        f"[Frame {idx}] "
        f"size={frame.width}x{frame.height} | "
        f"valid={valid_count}/{total} ({pct:.1f}%) | "
        f"Z range = {z_min:.1f}~{z_max:.1f} mm (median={z_med:.1f})"
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Output: {args.out}")
    print()

    with create_camera(cfg["camera"]) as cam:
        # 워밍업
        for i in range(args.warmup):
            _ = cam.capture()
            print(f"  warmup {i + 1}/{args.warmup} ...")

        # 본 캡처
        for i in range(args.num_frames):
            frame = cam.capture()
            print_stats(frame, i)
            save_frame(frame, args.out, i)

    print(f"\nDone. {args.num_frames} frame(s) saved to {args.out}")
    pcd = o3d.io.read_point_cloud('data/captures/test/frame_0000_pointcloud.ply')
    print(f'점 개수: {len(pcd.points)}')
    o3d.visualization.draw_geometries([pcd])

    return 0


if __name__ == "__main__":
    sys.exit(main())
