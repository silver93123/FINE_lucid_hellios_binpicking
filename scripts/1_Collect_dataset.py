"""Stage 2-1: 빈 피킹 데이터셋 수집 스크립트 (organized PCD 지원).

v2 변경점:
    - pointcloud_organized/ 폴더 추가: (H, W, 3) shape의 npy 저장
    - valid_mask/ 폴더 추가: (H, W) bool npy 저장
    - 기존 pointcloud/*.ply는 그대로 유지 (Open3D 호환)

용도:
    Detection 마스크와 PCD의 픽셀 매칭이 필요한 단계용.
    Stage 5 (crop_by_mask)에서 organized PCD가 필요함.

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/collect_dataset.py --out data/dataset/brackets_for_train --num 5 --start-index 11

저장 파일:
    {out_dir}/
    ├── intensity/frame_NNNN.png             ← 8-bit mono (RTMDet 입력)
    ├── pointcloud/frame_NNNN.ply            ← Open3D PCD (m 단위, valid만)
    ├── pointcloud_organized/frame_NNNN.npy  ← (H,W,3) mm 단위, NaN 포함
    ├── valid_mask/frame_NNNN.npy            ← (H,W) bool
    ├── metadata/frame_NNNN.json             ← 캡처 정보
    └── config_snapshot.yaml                 ← 카메라 설정 백업
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera import create_camera  # noqa: E402

num_capture = 300

def parse_args() -> argparse.Namespace:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="빈 피킹 데이터셋 수집 (organized PCD 포함)")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "dataset" / current_time)
    p.add_argument("--num", type=int, default=num_capture, help="캡처할 프레임 수")
    p.add_argument("--warmup", type=int, default=3, help="시작 시 버리는 워밍업 수")
    p.add_argument("--start-index", type=int, default=1, help="시작 프레임 번호")
    return p.parse_args()


def setup_output_dirs(out_dir: Path) -> dict:
    """출력 디렉토리 구조 생성."""
    subdirs = {
        "intensity": out_dir / "intensity",
        "pointcloud": out_dir / "pointcloud",
        "pointcloud_organized": out_dir / "pointcloud_organized",  # v2 추가
        "valid_mask": out_dir / "valid_mask",                       # v2 추가
        "metadata": out_dir / "metadata",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


def save_frame(frame, dirs: dict, idx: int, cfg_camera: dict) -> dict:
    """한 프레임을 저장 (4가지 형식)."""
    name = f"frame_{idx:04d}"

    # 1. Intensity PNG
    cv2.imwrite(str(dirs["intensity"] / f"{name}.png"), frame.intensity)

    # 2. Organized PCD (mm 단위, NaN 포함) - Stage 5용 핵심
    np.save(dirs["pointcloud_organized"] / f"{name}.npy",
            frame.points_organized.astype(np.float32))

    # 3. Valid mask
    np.save(dirs["valid_mask"] / f"{name}.npy",
            frame.valid_mask.astype(bool))

    # 4. Open3D PLY (m 단위, valid points만) - 기존 호환성 유지
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame.points / 1000.0)
        ok = o3d.io.write_point_cloud(
            str(dirs["pointcloud"] / f"{name}.ply"), pcd, write_ascii=False
        )
        if not ok:
            print(f"  [WARN] PLY 쓰기 실패", flush=True)
    except ImportError:
        pass  # Open3D 없으면 npy만으로 OK

    # 통계
    valid_count = int(frame.valid_mask.sum())
    total = frame.height * frame.width
    valid_pct = 100.0 * valid_count / total

    pts = frame.points
    if pts.size > 0:
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        z_med = float(np.median(pts[:, 2]))
    else:
        z_min = z_max = z_med = float("nan")

    metadata = {
        "frame_index": idx,
        "frame_name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": {"width": int(frame.width), "height": int(frame.height)},
        "stats": {
            "valid_pixels": valid_count,
            "total_pixels": total,
            "valid_ratio": round(valid_pct, 2),
            "z_min_mm": round(z_min, 1) if not np.isnan(z_min) else None,
            "z_max_mm": round(z_max, 1) if not np.isnan(z_max) else None,
            "z_median_mm": round(z_med, 1) if not np.isnan(z_med) else None,
            "num_points": int(len(pts)),
        },
        "camera_config": {
            "type": cfg_camera.get("type"),
            "pixel_format": cfg_camera.get("pixel_format"),
            "exposure_time_selector": cfg_camera.get("exposure_time_selector"),
            "operating_mode": cfg_camera.get("operating_mode"),
        },
        "files": {
            "intensity": f"intensity/{name}.png",
            "pointcloud": f"pointcloud/{name}.ply",
            "pointcloud_organized": f"pointcloud_organized/{name}.npy",
            "valid_mask": f"valid_mask/{name}.npy",
        },
    }

    with (dirs["metadata"] / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dirs = setup_output_dirs(args.out)
    shutil.copy2(args.config, args.out / "config_snapshot.yaml")

    print("=" * 70, flush=True)
    print("  데이터셋 수집 (v2: organized PCD 포함)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Config:    {args.config}", flush=True)
    print(f"  Output:    {args.out}", flush=True)
    print(f"  Frames:    {args.start_index} ~ {args.start_index + args.num - 1}", flush=True)
    print(f"  Camera:    exposure={cfg['camera']['exposure_time_selector']}, "
          f"mode={cfg['camera']['operating_mode']}", flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)
    print("  부품 배치를 매 프레임마다 바꿔주세요.", flush=True)
    print("  Stage 5 검증용: organized PCD와 valid_mask도 함께 저장됩니다.", flush=True)
    print("", flush=True)

    captured = []
    try:
        with create_camera(cfg["camera"]) as cam:
            print(f"카메라 워밍업 ({args.warmup} frames)...", flush=True)
            for i in range(args.warmup):
                _ = cam.capture()
                print(f"  {i + 1}/{args.warmup}", flush=True)
            print("", flush=True)

            for k in range(args.num):
                idx = args.start_index + k
                print("-" * 70, flush=True)
                print(f"[{k + 1}/{args.num}] 프레임 {idx:04d}", flush=True)
                print(f"  → 배치 후 Enter (q=종료, s=스킵)", flush=True)

                try:
                    user_input = input("  > ").strip().lower()
                except KeyboardInterrupt:
                    print("\n  중단됨.", flush=True)
                    break

                if user_input == "q":
                    break
                if user_input == "s":
                    continue

                t0 = time.perf_counter()
                frame = cam.capture()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                meta = save_frame(frame, dirs, idx, cfg["camera"])

                s = meta["stats"]
                print(f"  ✓ saved | {dt_ms:5.1f} ms | "
                      f"valid {s['valid_ratio']:.1f}% | "
                      f"Z {s['z_min_mm']}~{s['z_max_mm']} mm "
                      f"(median {s['z_median_mm']})", flush=True)
                captured.append(meta)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70, flush=True)
    print(f"  완료: {len(captured)} 프레임", flush=True)
    print("=" * 70, flush=True)
    print(f"  저장: {args.out}", flush=True)

    if captured:
        valid_ratios = [m["stats"]["valid_ratio"] for m in captured]
        print(f"\n  valid: mean={np.mean(valid_ratios):.1f}%, "
              f"min={min(valid_ratios):.1f}%, max={max(valid_ratios):.1f}%",
              flush=True)
        print(f"\n  organized PCD: {args.out / 'pointcloud_organized'}", flush=True)
        print(f"  valid mask:    {args.out / 'valid_mask'}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())