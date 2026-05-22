"""실전 빈피킹 파이프라인: 카메라 캡처 → Detection → ICP → 픽포인트 산출.

흐름:
    [1] 카메라 초기화 + 워밍업
    [2] RTMDet 모델 로드
    [3] CAD 모델 로드
    [4] 캡처 루프 (Enter → 캡처 → 저장 → Detection → ICP → 픽포인트 출력)

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/run_binpicking.py

    옵션:
        --config  config/config.yaml   카메라 설정 파일 (기본값)
        --warmup  3                    워밍업 프레임 수
        --out     data/captures/live   캡처 저장 경로

출력 (캡처마다 갱신):
    data/captures/live/
    ├── intensity/frame_NNNN.png
    ├── pointcloud_organized/frame_NNNN.npy
    ├── valid_mask/frame_NNNN.npy
    ├── metadata/frame_NNNN.json
    └── results/
        ├── frame_NNNN_overlay.png
        ├── frame_NNNN_colored.ply
        ├── frame_NNNN_obj{i}.ply
        ├── frame_NNNN_summary.json
        ├── frame_NNNN_obj{i}_icp_vis.ply
        └── frame_NNNN_obj{i}_pose.json

조작:
    Enter  → 캡처 + 픽포인트 산출
    q      → 종료
"""

from __future__ import annotations

import argparse
import copy
import json
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

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d 필요. pip install open3d", flush=True)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera import create_camera        # noqa: E402
from src.detection import RTMDetInferencer  # noqa: E402


# =============================================================================
# 설정
# =============================================================================

# ── Detection ─────────────────────────────────────────────────────────────────
WORK_DIR        = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1"
CONFIG_PATH     = WORK_DIR / "rtmdet-ins_bracket.py"

# best_*.pth 자동 탐색 (epoch 수 무관)
_candidates = sorted(WORK_DIR.glob("best_*.pth"))
if not _candidates:
    print(f"ERROR: best 모델이 없습니다: {WORK_DIR}", flush=True)
    sys.exit(1)
CHECKPOINT_PATH = _candidates[-1] 

SCORE_THRESHOLD         = 0.3
MIN_POINTS_PER_INSTANCE = 100

# ── ICP ───────────────────────────────────────────────────────────────────────
CAD_PATH = ROOT / "data" / "cad" / "bracket_v2.stl"

CAD_SAMPLE_POINTS = 20000
VOXEL_SIZE_CAD    = 0.002   # 2mm
VOXEL_SIZE_SCENE  = 0.003   # 3mm

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

ICP_STAGES = [
    {"max_dist": 0.020, "max_iter": 100},   # 20mm — 초기 오차 흡수
    {"max_dist": 0.010, "max_iter": 100},   # 10mm — 중간 수렴
    {"max_dist": 0.005, "max_iter": 100},   # 5mm  — 정밀 수렴
]
ICP_FITNESS_THRESHOLD = 0.5
XYZ_MAX_M             = 2.0

# 제품정렬 상태에 따른 CAD 회전축 보정 (Rx=-90, Ry=90, Rz=90 — 시각적으로 확인된 값)
CAD_AXIS_CORRECTION_DEG = (-90, 90, 90)

# ── 픽포인트 ──────────────────────────────────────────────────────────────────
# 축 보정 후 CAD 로컬 좌표계의 상단 수평면 중심 (단위: m)
CAD_PICK_LOCAL = np.array([0.000, -0.100, 0.031, 1.0])

# 미세 조정 오프셋 (CAD 로컬 기준, 단위: mm)
# X: 브라켓 폭 방향   (+: 오른쪽, -: 왼쪽)
# Y: 브라켓 길이 방향  (+: 앞,    -: 뒤)
# Z: 브라켓 높이 방향  (+: 위,    -: 아래)
PICK_OFFSET_X_MM = -5.0
PICK_OFFSET_Y_MM =  0.0
PICK_OFFSET_Z_MM =  0.0

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
_PALETTE_BGR = np.array([
    [ 50,  50, 255], [ 50, 200,  50], [255, 100,  50],
    [ 30, 180, 255], [230,  50, 180], [200, 200,  30],
], dtype=np.uint8)
_PALETTE_RGB_FLOAT = _PALETTE_BGR[:, ::-1].astype(np.float64) / 255.0
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)


# =============================================================================
# 카메라 캡처 + 저장
# =============================================================================

def setup_dirs(out_dir: Path) -> dict[str, Path]:
    """출력 디렉토리 구조 생성."""
    subdirs = {
        "intensity":            out_dir / "intensity",
        "pointcloud_organized": out_dir / "pointcloud_organized",
        "valid_mask":           out_dir / "valid_mask",
        "metadata":             out_dir / "metadata",
        "results":              out_dir / "results",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


def save_capture(frame, dirs: dict, idx: int, cfg_camera: dict) -> dict:
    """캡처 프레임을 intensity / organized PCD / valid_mask / metadata로 저장.

    Returns:
        metadata dict (통계 포함)
    """
    name = f"frame_{idx:04d}"

    # intensity PNG
    cv2.imwrite(str(dirs["intensity"] / f"{name}.png"), frame.intensity)

    # organized PCD (mm 단위, NaN 포함)
    np.save(dirs["pointcloud_organized"] / f"{name}.npy",
            frame.points_organized.astype(np.float32))

    # valid mask
    np.save(dirs["valid_mask"] / f"{name}.npy",
            frame.valid_mask.astype(bool))

    # 통계
    pts       = frame.points
    valid_cnt = int(frame.valid_mask.sum())
    total     = frame.height * frame.width

    if pts.size > 0:
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        z_med = float(np.median(pts[:, 2]))
    else:
        z_min = z_max = z_med = float("nan")

    metadata = {
        "frame_index": idx,
        "frame_name":  name,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "image":       {"width": int(frame.width), "height": int(frame.height)},
        "stats": {
            "valid_pixels": valid_cnt,
            "total_pixels": total,
            "valid_ratio":  round(100.0 * valid_cnt / total, 2),
            "z_min_mm":     round(z_min, 1) if not np.isnan(z_min) else None,
            "z_max_mm":     round(z_max, 1) if not np.isnan(z_max) else None,
            "z_median_mm":  round(z_med, 1) if not np.isnan(z_med) else None,
            "num_points":   int(len(pts)),
        },
        "files": {
            "intensity":            f"intensity/{name}.png",
            "pointcloud_organized": f"pointcloud_organized/{name}.npy",
            "valid_mask":           f"valid_mask/{name}.npy",
        },
    }
    with (dirs["metadata"] / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


# =============================================================================
# Detection + PCD 분리
# =============================================================================

def overlay_results(image_bgr: np.ndarray, results, valid_mask=None) -> np.ndarray:
    """Detection 결과를 2D 이미지에 시각화."""
    overlay = image_bgr.copy()
    if valid_mask is not None:
        overlay[~valid_mask] = (overlay[~valid_mask] * 0.4).astype(np.uint8)
    for i, r in enumerate(results):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        layer = np.zeros_like(overlay)
        layer[r.mask] = color
        overlay[r.mask] = (0.5 * overlay[r.mask] + 0.5 * layer[r.mask]).astype(np.uint8)
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


def save_instance_pcd(points: np.ndarray, out_path: Path, color: tuple) -> bool:
    if points.size == 0:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    )
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def save_colored_full_pcd(
    pcd_organized: np.ndarray,
    valid_mask: np.ndarray,
    results,
    out_path: Path,
) -> bool:
    all_pts = pcd_organized[valid_mask]
    if len(all_pts) == 0:
        return False
    colors = np.tile(_BG_COLOR, (len(all_pts), 1))
    H, W = valid_mask.shape
    lookup = np.full((H, W), -1, dtype=np.int32)
    vr, vc = np.where(valid_mask)
    lookup[vr, vc] = np.arange(len(vr))
    for i, r in enumerate(results):
        ir, ic = np.where(r.mask & valid_mask)
        if len(ir):
            colors[lookup[ir, ic]] = _PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))


def run_detection(
    frame_name: str,
    gray: np.ndarray,
    pcd_organized: np.ndarray,
    valid_mask: np.ndarray,
    inferencer: RTMDetInferencer,
    result_dir: Path,
) -> tuple[dict, list[Path]]:
    """Detection + PCD 분리 실행. (summary dict, 인스턴스 PLY 경로 목록) 반환."""
    H, W = gray.shape
    bgr  = np.stack([gray, gray, gray], axis=-1)

    results = inferencer.infer(bgr)

    # 2D overlay
    cv2.imwrite(str(result_dir / f"{frame_name}_overlay.png"),
                overlay_results(bgr, results, valid_mask))

    # 전체 colored PCD
    colored_path = result_dir / f"{frame_name}_colored.ply"
    colored_ok   = save_colored_full_pcd(pcd_organized, valid_mask, results, colored_path)

    instances_info = []
    instance_plys  = []

    for i, r in enumerate(results):
        combined   = r.mask & valid_mask
        obj_pts    = pcd_organized[combined]

        if len(obj_pts) < MIN_POINTS_PER_INSTANCE:
            instances_info.append({
                "instance_id": i, "class": r.class_name,
                "score": float(r.score),
                "skipped": "점이 너무 적음", "num_points": int(len(obj_pts)),
            })
            continue

        color_rgb = tuple(_PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)].tolist())
        ply_path  = result_dir / f"{frame_name}_obj{i}.ply"
        ok        = save_instance_pcd(obj_pts, ply_path, color=color_rgb)

        center    = obj_pts.mean(axis=0)
        size      = obj_pts.max(axis=0) - obj_pts.min(axis=0)

        instances_info.append({
            "instance_id": i,
            "class": r.class_name,
            "score": float(r.score),
            "num_pixels_detection":  int(r.mask.sum()),
            "num_pixels_after_valid": int(combined.sum()),
            "num_points_3d": int(len(obj_pts)),
            "valid_overlap_ratio": float(combined.sum()) / max(int(r.mask.sum()), 1),
            "bbox_2d":     r.bbox.tolist(),
            "center_mm":   center.tolist(),
            "size_mm":     size.tolist(),
            "z_median_mm": float(np.median(obj_pts[:, 2])),
            "ply_saved":   ok,
            "ply_path":    str(ply_path.relative_to(ROOT)) if ok else None,
        })
        if ok:
            instance_plys.append(ply_path)

    summary = {
        "frame": frame_name,
        "image_size": [int(H), int(W)],
        "valid_mask_ratio":  float(valid_mask.mean()),
        "num_detected":      len(results),
        "num_with_pcd":      len([x for x in instances_info if "skipped" not in x]),
        "colored_ply_saved": colored_ok,
        "instances":         instances_info,
    }
    with (result_dir / f"{frame_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, instance_plys


# =============================================================================
# ICP + 픽포인트
# =============================================================================

def _Rx(d):
    a = np.radians(d)
    return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

def _Ry(d):
    a = np.radians(d)
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])

def _Rz(d):
    a = np.radians(d)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])


def load_cad_as_pcd(stl_path: Path) -> o3d.geometry.PointCloud:
    """STL → mm→m 변환 → 축 보정 → 포인트 샘플링."""
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if not mesh.has_triangles():
        raise ValueError(f"STL 로드 실패: {stl_path}")

    ext_before = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    STL 원본 extent: {np.round(ext_before, 2)} mm", flush=True)
    mesh.scale(1.0 / 1000.0, center=np.zeros(3))   # 원점 기준 스케일
    ext_after = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    변환 후 extent:  {np.round(ext_after, 4)} m  "
          f"center={np.round(np.asarray(mesh.get_center()), 4)}", flush=True)

    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R      = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix  = np.eye(4)
    T_fix[:3, :3] = R
    T_fix[:3, 3]  = center - R @ center
    mesh.transform(T_fix)
    print(f"    축 보정: Rx={rx}° Ry={ry}° Rz={rz}°", flush=True)

    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)


def run_icp_multistage(src, tgt, T_init):
    T = T_init.copy()
    for stage in ICP_STAGES:
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, stage["max_dist"], T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=stage["max_iter"]),
        )
        T = np.asarray(res.transformation)
    final = o3d.pipelines.registration.evaluate_registration(
        src, tgt, ICP_STAGES[-1]["max_dist"], T)
    return T, float(final.fitness), float(final.inlier_rmse)


def correct_flipped_pose(T, src, tgt):
    if T[:3, :3][2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            src, tgt, ICP_STAGES[-1]["max_dist"], T)
        return T, float(final.fitness), float(final.inlier_rmse), False
    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4)
    T_flip[:3, :3] = R_flip
    c = T[:3, 3]
    T_flip[:3, 3]  = c - R_flip @ c
    T_f, fit, rmse = run_icp_multistage(src, tgt, T_flip @ T)
    return T_f, fit, rmse, True


def transform_to_pose(T):
    xyz_mm = (T[:3, 3] * 1000.0).tolist()
    R = T[:3, :3]
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2,1]/cp, R[2,2]/cp)
        yaw  = np.arctan2(R[1,0]/cp, R[0,0]/cp)
    else:
        roll, yaw = 0.0, np.arctan2(-R[0,1], R[1,1])
    e = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm": [round(v, 3) for v in xyz_mm],
        "euler_deg": {"roll_deg": round(e[0],4), "pitch_deg": round(e[1],4), "yaw_deg": round(e[2],4)},
        "transform_matrix": T.tolist(),
    }


def compute_pick_point(T):
    pl = CAD_PICK_LOCAL.copy()
    pl[0] += PICK_OFFSET_X_MM / 1000.0
    pl[1] += PICK_OFFSET_Y_MM / 1000.0
    pl[2] += PICK_OFFSET_Z_MM / 1000.0
    wt  = T @ pl
    pos = (wt[:3] * 1000.0).tolist()
    app = T[:3, 2] / (np.linalg.norm(T[:3, 2]) + 1e-9)
    R   = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2,0], np.sqrt(R[0,0]**2+R[1,0]**2))))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R[2,1]/cp, R[2,2]/cp)))
        yaw  = float(np.degrees(np.arctan2(R[1,0]/cp, R[0,0]/cp)))
    else:
        roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0,1], R[1,1])))
    return {
        "position_mm":  [round(v, 3) for v in pos],
        "approach_vec": [round(v, 6) for v in app.tolist()],
        "approach_deg": {"roll_deg": round(roll,4), "pitch_deg": round(pitch,4), "yaw_deg": round(yaw,4)},
    }


def save_icp_visualization(scene_pcd, cad_pcd, T, pick, out_path):
    sv = copy.deepcopy(scene_pcd)
    n  = len(np.asarray(sv.points))
    sv.colors = o3d.utility.Vector3dVector(np.tile([0.6,0.6,0.6], (n,1)))

    cv = copy.deepcopy(cad_pcd)
    cv.transform(T)
    n  = len(np.asarray(cv.points))
    cv.colors = o3d.utility.Vector3dVector(np.tile([0.1,0.9,0.3], (n,1)))

    pm = np.array(pick["position_mm"]) / 1000.0
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sp.translate(pm)
    sp.paint_uniform_color([1.0, 0.1, 0.1])
    sp_pcd = sp.sample_points_uniformly(500)

    app = np.array(pick["approach_vec"])
    ap  = np.array([pm + t * app * 0.03 for t in np.linspace(0,1,50)])
    ap_pcd = o3d.geometry.PointCloud()
    ap_pcd.points = o3d.utility.Vector3dVector(ap)
    ap_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1,0.3,1.0], (50,1)))

    o3d.io.write_point_cloud(str(out_path), sv + cv + sp_pcd + ap_pcd, write_ascii=False)


def run_icp_for_frame(
    instance_plys: list[Path],
    cad_pcd: o3d.geometry.PointCloud,
    cad_down: o3d.geometry.PointCloud,
    result_dir: Path,
) -> list[dict]:
    """프레임의 모든 인스턴스에 대해 ICP + 픽포인트 계산."""
    icp_results = []

    for ply_path in instance_plys:
        stem      = ply_path.stem
        scene_pcd = o3d.io.read_point_cloud(str(ply_path))
        n_pts     = len(np.asarray(scene_pcd.points))
        if n_pts < 50:
            icp_results.append({"file": stem, "error": f"포인트 부족: {n_pts}개"})
            continue

        print(f"  {stem}: {n_pts} pts", flush=True)

        # 전처리
        sc, _    = scene_pcd.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO)
        n_after  = len(np.asarray(sc.points))
        sd       = sc.voxel_down_sample(VOXEL_SIZE_SCENE)
        rem_pct  = (1 - n_after / max(n_pts, 1)) * 100
        print(f"    노이즈 제거: {n_pts} → {n_after} pts ({rem_pct:.1f}%)", flush=True)
        print(f"    다운샘플:    scene={len(np.asarray(sd.points))}  cad={len(np.asarray(cad_down.points))}", flush=True)

        # 중심 정렬
        T_init = np.eye(4)
        T_init[:3, 3] = np.asarray(sd.get_center()) - np.asarray(cad_down.get_center())
        print(f"    중심 정렬:   {np.round(np.asarray(cad_down.get_center()),3)} → "
              f"{np.round(np.asarray(sd.get_center()),3)}", flush=True)

        # ICP
        print(f"    ICP 중...", flush=True)
        T, fit, rmse = run_icp_multistage(cad_down, sd, T_init)
        print(f"    ICP fitness={fit:.4f}, rmse={rmse:.6f}", flush=True)

        # 뒤집힘 보정
        T, fit, rmse, flipped = correct_flipped_pose(T, cad_down, sd)
        if flipped:
            print(f"    △ 뒤집힘 보정 후 fitness={fit:.4f}", flush=True)

        t_mm = np.round(T[:3, 3] * 1000, 1)
        print(f"    T translation: {t_mm} mm", flush=True)

        if fit < ICP_FITNESS_THRESHOLD:
            print(f"    ✗ ICP 실패 (fitness={fit:.4f})", flush=True)
            icp_results.append({"file": stem, "error": "ICP 정합 실패",
                                 "icp_fitness": float(fit)})
            continue

        if max(abs(v) for v in T[:3, 3]) > XYZ_MAX_M:
            print(f"    ✗ xyz 비정상: {t_mm} mm", flush=True)
            icp_results.append({"file": stem, "error": "xyz 범위 이상",
                                 "icp_fitness": float(fit)})
            continue

        pose = transform_to_pose(T)
        pick = compute_pick_point(T)
        xyz  = pose["xyz_mm"]
        eul  = pose["euler_deg"]
        ppos = pick["position_mm"]
        avec = pick["approach_vec"]
        adeg = pick["approach_deg"]

        vis_path = result_dir / f"{stem}_icp_vis.ply"
        save_icp_visualization(scene_pcd, cad_pcd, T, pick, vis_path)

        result = {
            "file": stem,
            "num_points_scene": n_pts,
            "num_points_after_outlier_removal": n_after,
            "icp_fitness": float(fit),
            "icp_rmse_m":  float(rmse),
            "was_flipped": flipped,
            "pose":        pose,
            "pick_point":  pick,
            "vis_ply":     str(vis_path.relative_to(ROOT)),
        }
        with (result_dir / f"{stem}_pose.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"    ✓ CAD 중심:  xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm  "
              f"roll={eul['roll_deg']:.1f}° pitch={eul['pitch_deg']:.1f}° "
              f"yaw={eul['yaw_deg']:.1f}°", flush=True)
        print(f"    ✓ 픽포인트:  xyz=({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm", flush=True)
        print(f"    ✓ 접근 방향: vec=({avec[0]:.3f}, {avec[1]:.3f}, {avec[2]:.3f})  "
              f"roll={adeg['roll_deg']:.1f}° pitch={adeg['pitch_deg']:.1f}° "
              f"yaw={adeg['yaw_deg']:.1f}°", flush=True)

        icp_results.append(result)

    return icp_results


# =============================================================================
# 메인
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실전 빈피킹 파이프라인")
    p.add_argument("--config",  type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--out",     type=Path, default=ROOT / "data" / "captures" / "live")
    p.add_argument("--warmup",  type=int,  default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Camera 설정 파일 로드
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 사전 점검
    for path, name in [
        (CONFIG_PATH,     "RTMDet config"),
        (CHECKPOINT_PATH, "RTMDet checkpoint"),
        (CAD_PATH,        "CAD STL"),
        (args.config,     "camera config"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} 없음: {path}", flush=True)
            return 1

    dirs = setup_dirs(args.out)

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    print("=" * 70, flush=True)
    print("  실전 빈피킹 파이프라인", flush=True)
    print("=" * 70, flush=True)
    print(f"  Config:      {args.config}", flush=True)
    print(f"  Output:      {args.out}", flush=True)
    print(f"  CAD:         {CAD_PATH.name}", flush=True)
    print(f"  축 보정:     Rx={CAD_AXIS_CORRECTION_DEG[0]}° "
          f"Ry={CAD_AXIS_CORRECTION_DEG[1]}° Rz={CAD_AXIS_CORRECTION_DEG[2]}°", flush=True)
    stages_str = " → ".join(f"{s['max_dist']*1000:.0f}mm×{s['max_iter']}" for s in ICP_STAGES)
    print(f"  ICP 단계:    {stages_str}", flush=True)
    print(f"  픽 오프셋:   X={PICK_OFFSET_X_MM:+.1f}mm  "
          f"Y={PICK_OFFSET_Y_MM:+.1f}mm  Z={PICK_OFFSET_Z_MM:+.1f}mm", flush=True)
    print("=" * 70, flush=True)

    # ── [1] 모델 로드 ─────────────────────────────────────────────────────────
    print("\n[1] RTMDet 모델 로드 중...", flush=True)
    inferencer = RTMDetInferencer(
        config=CONFIG_PATH,
        checkpoint=CHECKPOINT_PATH,
        device="cuda:0",
        score_threshold=SCORE_THRESHOLD,
    )
    print(f"    ✓ 클래스: {inferencer.class_names}", flush=True)

    print("\n[2] CAD 모델 로드 중...", flush=True)
    try:
        cad_pcd = load_cad_as_pcd(CAD_PATH)
    except Exception as e:
        print(f"ERROR: CAD 로드 실패: {e}", flush=True)
        return 1
    cad_down = cad_pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    print(f"    ✓ CAD 샘플: {len(np.asarray(cad_pcd.points))}pts  "
          f"다운샘플: {len(np.asarray(cad_down.points))}pts", flush=True)

    # ── [2] 카메라 초기화 ─────────────────────────────────────────────────────
    print(f"\n[3] 카메라 초기화 + 워밍업 ({args.warmup} frames)...", flush=True)
    frame_idx = 0

    try:
        with create_camera(cfg["camera"]) as cam:

            for i in range(args.warmup):
                _ = cam.capture()
                print(f"    워밍업 {i+1}/{args.warmup}", flush=True)
            print("    ✓ 카메라 준비 완료", flush=True)

            # ── [3] 캡처 루프 ────────────────────────────────────────────────
            print("\n" + "=" * 70, flush=True)
            print("  캡처 루프 시작", flush=True)
            print("  Enter → 캡처 + 픽포인트 산출  |  q → 종료", flush=True)
            print("=" * 70, flush=True)

            while True:
                try:
                    user_input = input("\n  > ").strip().lower()
                except KeyboardInterrupt:
                    print("\n  중단됨.", flush=True)
                    break

                if user_input == "q":
                    print("  종료.", flush=True)
                    break

                frame_idx += 1
                frame_name = f"frame_{frame_idx:04d}"
                print(f"\n{'─'*70}", flush=True)
                print(f"  [{frame_name}] 캡처 중...", flush=True)

                # 캡처
                t0    = time.perf_counter()
                frame = cam.capture()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                # 저장
                meta = save_capture(frame, dirs, frame_idx, cfg["camera"])
                s    = meta["stats"]
                print(f"  캡처 완료: {dt_ms:.1f} ms | "
                      f"valid {s['valid_ratio']:.1f}% | "
                      f"Z {s['z_min_mm']}~{s['z_max_mm']} mm", flush=True)

                # intensity + organized PCD + valid mask 로드
                gray         = frame.intensity
                pcd_organized = frame.points_organized.astype(np.float32)
                valid_mask   = frame.valid_mask.astype(bool)

                # Detection + PCD 분리
                print(f"\n  [Detection]", flush=True)
                t0 = time.perf_counter()
                summary, inst_plys = run_detection(
                    frame_name, gray, pcd_organized, valid_mask,
                    inferencer, dirs["results"]
                )
                det_ms = (time.perf_counter() - t0) * 1000.0

                n_det = summary["num_detected"]
                n_pcd = summary["num_with_pcd"]
                print(f"  검출: {n_det}개  PCD 추출: {n_pcd}개  ({det_ms:.0f} ms)", flush=True)

                for inst in summary["instances"]:
                    if "skipped" in inst:
                        print(f"    #{inst['instance_id']}: SKIPPED "
                              f"({inst['skipped']}, {inst['num_points']} pts)", flush=True)
                        continue
                    c = inst["center_mm"]
                    print(f"    #{inst['instance_id']}: score={inst['score']:.2f}  "
                          f"{inst['num_points_3d']} pts  "
                          f"center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm", flush=True)

                if not inst_plys:
                    print("  검출된 브라켓 없음 — 다시 시도하세요.", flush=True)
                    continue

                # ICP + 픽포인트
                print(f"\n  [ICP + 픽포인트]", flush=True)
                t0 = time.perf_counter()
                icp_results = run_icp_for_frame(inst_plys, cad_pcd, cad_down, dirs["results"])
                icp_ms = (time.perf_counter() - t0) * 1000.0

                n_ok   = sum(1 for r in icp_results if "error" not in r)
                n_fail = len(icp_results) - n_ok
                print(f"\n  ICP 완료: 성공 {n_ok}개  실패 {n_fail}개  ({icp_ms:.0f} ms)",
                      flush=True)

                # 픽포인트 최종 출력 (성공한 것만)
                success = [r for r in icp_results if "error" not in r]
                if success:
                    print(f"\n  ┌{'─'*50}", flush=True)
                    print(f"  │ 픽포인트 목록 ({len(success)}개)", flush=True)
                    print(f"  ├{'─'*50}", flush=True)
                    for r in success:
                        pp = r["pick_point"]["position_mm"]
                        av = r["pick_point"]["approach_vec"]
                        print(f"  │  {r['file']}", flush=True)
                        print(f"  │    위치:     ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm",
                              flush=True)
                        print(f"  │    접근벡터: ({av[0]:.3f}, {av[1]:.3f}, {av[2]:.3f})",
                              flush=True)
                    print(f"  └{'─'*50}", flush=True)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n총 {frame_idx}회 캡처 완료. 결과: {args.out / 'results'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())