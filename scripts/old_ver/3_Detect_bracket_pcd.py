"""Stage 5-A: 학습된 RTMDet-Ins 추론 + organized PCD 결합 → 객체별 PCD 분리.

용도:
    Detection mask를 organized PCD에 적용해서 각 객체(브라켓)의 3D 점들만 분리.
    이게 빈 피킹의 "어떤 부품이 어디에 있는가" 단계.

전제 조건:
    - 학습된 모델: work_dirs/rtmdet-ins_bracket_v1/best_coco_bbox_mAP_epoch_50.pth
    - 새 캡처 데이터: data/dataset/brackets_for_train/ (v2, organized PCD 포함)

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/6_stage5_crop_pcd.py

출력:
    data/inference_results/stage5/
    ├── frame_NNNN_overlay.png      ← detection 결과 시각화
    ├── frame_NNNN_obj0.ply         ← 0번 인스턴스 PCD (단독)
    ├── frame_NNNN_obj1.ply         ← 1번 인스턴스 PCD (단독)
    ├── frame_NNNN_colored.ply      ← 전체 PCD + 인스턴스별 색상 (Open3D 시각화용)
    ├── frame_NNNN_summary.json     ← 인스턴스별 통계
    └── ...

다음 단계 (Stage 5-B):
    각 obj{i}.ply에 CAD ICP 정합 → 6DoF 자세 추정.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import RTMDetInferencer  # noqa: E402

input_data = "20260519_0000"

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
WORK_DIR = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1"
CONFIG_PATH = WORK_DIR / "rtmdet-ins_bracket.py"
CHECKPOINT_PATH = WORK_DIR / "best_coco_bbox_mAP_epoch_50.pth"

DATASET_DIR = ROOT / "data" / "dataset_input" / input_data
INTENSITY_DIR = DATASET_DIR / "intensity"
PCD_ORGANIZED_DIR = DATASET_DIR / "pointcloud_organized"
VALID_MASK_DIR = DATASET_DIR / "valid_mask"

OUTPUT_DIR = ROOT / "data" / "inference_results" / input_data

# 추론 임계값 (학습 안 한 데이터라 적당히 낮춤)
SCORE_THRESHOLD = 0.3

# PCD 후처리
MIN_POINTS_PER_INSTANCE = 100  # 너무 적은 점의 인스턴스는 노이즈로 간주

# 인스턴스 색상 팔레트 (2D overlay용 BGR, 3D PLY용 RGB float 공유)
# 순서: [BGR uint8 for OpenCV, RGB float for Open3D]
_PALETTE_BGR = np.array([
    [50,  50,  255],   # 빨강
    [50,  200,  50],   # 초록
    [255, 100,  50],   # 청록
    [30,  180, 255],   # 주황
    [230,  50, 180],   # 자홍
    [200, 200,  30],   # 노랑
], dtype=np.uint8)

# Open3D용: BGR → RGB, 0~255 → 0.0~1.0
_PALETTE_RGB_FLOAT = _PALETTE_BGR[:, ::-1].astype(np.float64) / 255.0

# 배경(검출 안 된 valid 포인트) 색상
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)  # 회색


# -----------------------------------------------------------------------------
# 시각화
# -----------------------------------------------------------------------------

def overlay_results(image_bgr: np.ndarray, results, valid_mask=None) -> np.ndarray:
    """Detection 결과 + (선택) valid_mask를 시각화."""
    overlay = image_bgr.copy()

    # 1. invalid 영역을 어둡게
    if valid_mask is not None:
        overlay[~valid_mask] = (overlay[~valid_mask] * 0.4).astype(np.uint8)

    # 2. 각 인스턴스 마스크
    for i, r in enumerate(results):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        color_layer = np.zeros_like(overlay)
        color_layer[r.mask] = color
        overlay[r.mask] = (0.5 * overlay[r.mask] + 0.5 * color_layer[r.mask]).astype(np.uint8)

    # 3. BBox + 라벨
    for i, r in enumerate(results):
        color = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
        x1, y1, x2, y2 = r.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"#{i} {r.class_name} {r.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


# -----------------------------------------------------------------------------
# PCD 처리
# -----------------------------------------------------------------------------

def crop_pcd_by_mask(
    pcd_organized: np.ndarray,
    valid_mask: np.ndarray,
    instance_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detection 마스크와 valid_mask를 결합해서 객체 PCD 추출.

    Args:
        pcd_organized: (H, W, 3) float32, mm 단위, invalid=NaN
        valid_mask:    (H, W) bool, 카메라가 신뢰하는 픽셀
        instance_mask: (H, W) bool, RTMDet이 검출한 객체 영역

    Returns:
        object_points: (N, 3) float32, mm 단위
        combined_mask: (H, W) bool, 실제 사용된 픽셀
    """
    combined = instance_mask & valid_mask
    object_points = pcd_organized[combined]
    return object_points, combined


def save_pcd(points: np.ndarray, out_path: Path, color: tuple = (0.4, 0.8, 0.4)) -> bool:
    """인스턴스 단독 PCD를 PLY로 저장 (mm → m 변환)."""
    try:
        import open3d as o3d
    except ImportError:
        np.save(out_path.with_suffix(".npy"), points)
        return True

    if points.size == 0:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)  # mm → m
    colors = np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def save_colored_full_pcd(
    pcd_organized: np.ndarray,
    valid_mask: np.ndarray,
    results,
    out_path: Path,
) -> bool:
    """전체 포인트 클라우드를 단일 PLY로 저장.

    Open3D에서 열었을 때:
        - 회색 포인트: 배경 (검출되지 않은 valid 포인트)
        - 컬러 포인트: 탐지된 인스턴스 (인스턴스별 고유 색상)
        - invalid(NaN) 포인트: 제외

    Args:
        pcd_organized: (H, W, 3) float32, mm 단위, invalid=NaN
        valid_mask:    (H, W) bool
        results:       RTMDet 추론 결과 리스트
        out_path:      저장 경로 (.ply)

    Returns:
        저장 성공 여부
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  WARNING: open3d 없음 - colored PLY 저장 스킵", flush=True)
        return False

    # valid 포인트만 추출 (NaN 제외)
    all_points = pcd_organized[valid_mask]   # (N, 3) float32, mm
    if len(all_points) == 0:
        return False

    # 기본 색상: 회색 (배경)
    colors = np.tile(_BG_COLOR, (len(all_points), 1))  # (N, 3) float64

    # valid 픽셀 위치 → all_points 배열 내 인덱스로 변환하는 lookup table
    # lookup[row, col] = all_points 내 해당 포인트의 인덱스 (-1: invalid)
    H, W = valid_mask.shape
    lookup = np.full((H, W), -1, dtype=np.int32)
    valid_rows, valid_cols = np.where(valid_mask)
    lookup[valid_rows, valid_cols] = np.arange(len(valid_rows))

    # 인스턴스별로 색상 덮어쓰기
    for i, r in enumerate(results):
        color = _PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)]

        # 인스턴스 마스크 & valid_mask 교집합
        inst_rows, inst_cols = np.where(r.mask & valid_mask)
        if len(inst_rows) == 0:
            continue

        # lookup으로 all_points 내 인덱스 취득
        positions = lookup[inst_rows, inst_cols]  # 모두 >= 0 (교집합이므로)
        colors[positions] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points / 1000.0)  # mm → m
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ok = bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))
    return ok


# -----------------------------------------------------------------------------
# 프레임 처리
# -----------------------------------------------------------------------------

def process_frame(
    frame_name: str,
    inferencer: RTMDetInferencer,
    output_dir: Path,
) -> dict:
    """한 프레임의 detection + PCD crop 처리."""

    # === 입력 파일 로드 ===
    intensity_path = INTENSITY_DIR / f"{frame_name}.png"
    pcd_path = PCD_ORGANIZED_DIR / f"{frame_name}.npy"
    valid_mask_path = VALID_MASK_DIR / f"{frame_name}.npy"

    if not all(p.exists() for p in [intensity_path, pcd_path, valid_mask_path]):
        return {"error": "파일 없음", "frame": frame_name}

    gray = cv2.imread(str(intensity_path), cv2.IMREAD_GRAYSCALE)
    bgr = np.stack([gray, gray, gray], axis=-1)  # RTMDet 입력 형식
    pcd_organized = np.load(pcd_path)            # (H, W, 3), float32, NaN 포함
    valid_mask = np.load(valid_mask_path)        # (H, W), bool

    H, W = gray.shape

    # === Detection ===
    results = inferencer.infer(bgr)

    # === 2D 시각화 ===
    overlay = overlay_results(bgr, results, valid_mask)
    cv2.imwrite(str(output_dir / f"{frame_name}_overlay.png"), overlay)

    # === 전체 PCD (인스턴스 색상 포함) 저장 ===
    colored_ply_path = output_dir / f"{frame_name}_colored.ply"
    colored_ok = save_colored_full_pcd(pcd_organized, valid_mask, results, colored_ply_path)
    if colored_ok:
        print(f"    colored PLY 저장: {colored_ply_path.name}", flush=True)

    # === 각 인스턴스의 PCD crop (단독 저장) ===
    instances_info = []

    for i, r in enumerate(results):
        object_points, combined_mask = crop_pcd_by_mask(
            pcd_organized, valid_mask, r.mask
        )

        if len(object_points) < MIN_POINTS_PER_INSTANCE:
            instances_info.append({
                "instance_id": i,
                "class": r.class_name,
                "score": float(r.score),
                "skipped": "점이 너무 적음",
                "num_points": int(len(object_points)),
            })
            continue

        # 인스턴스 단독 PLY (기존 동작 유지)
        color_rgb = tuple(_PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)].tolist())
        ply_path = output_dir / f"{frame_name}_obj{i}.ply"
        ok = save_pcd(object_points, ply_path, color=color_rgb)

        # 통계
        center = object_points.mean(axis=0)
        pcd_range = object_points.max(axis=0) - object_points.min(axis=0)
        z_median = float(np.median(object_points[:, 2]))

        info = {
            "instance_id": i,
            "class": r.class_name,
            "score": float(r.score),
            "num_pixels_detection": int(r.mask.sum()),
            "num_pixels_after_valid": int(combined_mask.sum()),
            "num_points_3d": int(len(object_points)),
            "valid_overlap_ratio": float(combined_mask.sum()) / max(int(r.mask.sum()), 1),
            "bbox_2d": r.bbox.tolist(),
            "center_mm": center.tolist(),
            "size_mm": pcd_range.tolist(),
            "z_median_mm": z_median,
            "ply_saved": ok,
            "ply_path": str(ply_path.relative_to(ROOT)) if ok else None,
        }
        instances_info.append(info)

    summary = {
        "frame": frame_name,
        "image_size": [int(H), int(W)],
        "valid_mask_ratio": float(valid_mask.mean()),
        "num_detected": len(results),
        "num_with_pcd": len([x for x in instances_info if "skipped" not in x]),
        "colored_ply_saved": colored_ok,
        "colored_ply_path": str(colored_ply_path.relative_to(ROOT)) if colored_ok else None,
        "instances": instances_info,
    }

    with (output_dir / f"{frame_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------

def main() -> int:
    # 사전 점검
    for path, name in [
        (CONFIG_PATH, "config"),
        (CHECKPOINT_PATH, "checkpoint"),
        (INTENSITY_DIR, "intensity dir"),
        (PCD_ORGANIZED_DIR, "organized PCD dir"),
        (VALID_MASK_DIR, "valid mask dir"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} 없음: {path}", flush=True)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  Stage 5-A: Detection + PCD crop_by_mask", flush=True)
    print("=" * 70, flush=True)
    print(f"  Config:      {CONFIG_PATH}", flush=True)
    print(f"  Checkpoint:  {CHECKPOINT_PATH.name}", flush=True)
    print(f"  Dataset:     {DATASET_DIR}", flush=True)
    print(f"  Output:      {OUTPUT_DIR}", flush=True)
    print(f"  Threshold:   {SCORE_THRESHOLD}", flush=True)
    print("=" * 70, flush=True)

    # 모델 로드
    print("\n[1] 모델 로드 중...", flush=True)
    inferencer = RTMDetInferencer(
        config=CONFIG_PATH,
        checkpoint=CHECKPOINT_PATH,
        device="cuda:0",
        score_threshold=SCORE_THRESHOLD,
    )
    print(f"    ✓ 클래스: {inferencer.class_names}", flush=True)

    # 처리할 프레임 목록
    frames = sorted(f.stem for f in INTENSITY_DIR.glob("frame_*.png"))
    if not frames:
        print("ERROR: 프레임 없음", flush=True)
        return 1

    print(f"\n[2] 처리: {len(frames)} 프레임", flush=True)
    print("-" * 70, flush=True)

    all_summaries = []

    for fname in frames:
        summary = process_frame(fname, inferencer, OUTPUT_DIR)
        all_summaries.append(summary)

        if "error" in summary:
            print(f"  {fname}: ERROR - {summary['error']}", flush=True)
            continue

        n_det = summary["num_detected"]
        n_pcd = summary["num_with_pcd"]
        valid_pct = summary["valid_mask_ratio"] * 100
        print(f"  {fname}: detected={n_det}, with_pcd={n_pcd}, "
              f"frame_valid={valid_pct:.1f}%", flush=True)

        for inst in summary["instances"]:
            if "skipped" in inst:
                print(f"    #{inst['instance_id']}: SKIPPED ({inst['skipped']}, "
                      f"{inst['num_points']} pts)", flush=True)
                continue
            center = inst["center_mm"]
            size = inst["size_mm"]
            overlap = inst["valid_overlap_ratio"] * 100
            print(f"    #{inst['instance_id']}: score={inst['score']:.3f}, "
                  f"{inst['num_points_3d']} pts, "
                  f"valid_overlap={overlap:.1f}%, "
                  f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm, "
                  f"size=({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}) mm",
                  flush=True)

    # 종합 요약
    print("-" * 70, flush=True)
    print(f"\n[3] 요약", flush=True)
    total_det = sum(s.get("num_detected", 0) for s in all_summaries)
    total_pcd = sum(s.get("num_with_pcd", 0) for s in all_summaries)
    print(f"  처리 프레임:        {len(frames)}", flush=True)
    print(f"  총 검출:            {total_det}", flush=True)
    print(f"  PCD 추출 성공:      {total_pcd}", flush=True)

    print(f"\n  결과 위치: {OUTPUT_DIR}", flush=True)
    print(f"  - *_overlay.png:   2D detection 시각화", flush=True)
    print(f"  - *_colored.ply:   전체 PCD (배경 회색 + 인스턴스 컬러)", flush=True)
    print(f"  - *_obj{{i}}.ply:    인스턴스 단독 PCD (mm→m, 색칠됨)", flush=True)
    print(f"  - *_summary.json:  각 프레임의 통계", flush=True)
    print(f"\n  Open3D에서 확인:", flush=True)
    print(f"    python -c \"import open3d as o3d; o3d.visualization.draw_geometries("
          f"[o3d.io.read_point_cloud('{OUTPUT_DIR}/frame_0001_colored.ply')])\"", flush=True)
    print(f"\n  ✓ Stage 5-A 완료", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())