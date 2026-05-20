"""Stage 5-B: 인스턴스 PCD ↔ CAD 모델 ICP 정합 → 6DoF 자세 추정 (경량 버전).

파이프라인:
    1. STL CAD 모델 로드 → mm→m 변환 → 축 보정 회전
    2. 인스턴스 PCD 전처리 (노이즈 제거 → 다운샘플)
    3. CAD를 scene 중심으로 초기 정렬
    4. Point-to-Point ICP (노말 계산 불필요 → 빠름)
    5. 뒤집힘 감지 및 보정 (Z축 180도 회전 후 재수렴)
    6. 6DoF 자세 추출 + PLY 시각화 저장

전제 조건:
    브라켓이 항상 비슷한 방향(눕혀진 상태)으로 놓여있어야 함.
    무작위 방향으로 쌓이는 경우 RANSAC 버전 사용 필요.

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/7_stage5b_icp.py

입력:
    data/inference_results/{input_data}/frame_NNNN_obj{i}.ply
    data/cad/bracket_v2.stl

출력:
    data/inference_results/{input_data}/
    ├── frame_NNNN_obj{i}_pose.json
    └── frame_NNNN_obj{i}_icp_vis.ply
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

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

input_data = "20260519_0000"

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
CAD_PATH = ROOT / "data" / "cad" / "bracket_v2.stl"
INFERENCE_DIR = ROOT / "data" / "inference_results" / input_data
OUTPUT_DIR = INFERENCE_DIR

CAD_SAMPLE_POINTS = 20000       # RANSAC 버전보다 줄여도 됨 (FPFH 안 씀)

# 다운샘플링 복셀 크기 (m)
VOXEL_SIZE_CAD   = 0.002        # 2mm
VOXEL_SIZE_SCENE = 0.003        # 3mm

# 노이즈 제거
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

# ICP — Point-to-Point (노말 불필요)
# 단계적으로 거리를 좁혀가며 수렴 (coarse → fine)
ICP_STAGES = [
    {"max_dist": 0.020, "max_iter": 100},   # 20mm — 초기 정렬 오차 흡수
    {"max_dist": 0.010, "max_iter": 100},   # 10mm — 중간 수렴
    {"max_dist": 0.005, "max_iter": 100},   # 5mm  — 정밀 수렴
]
ICP_FITNESS_THRESHOLD = 0.5     # Point-to-Point는 기준을 높게 설정

# xyz 정상 범위 (m)
XYZ_MAX_M = 2.0

# CAD 축 보정 (Rx=-90, Ry=90, Rz=90 — 시각적으로 확인된 값)
CAD_AXIS_CORRECTION_DEG = (-90, 90, 90)

# 픽포인트 설정
# 축 보정(Rx=-90, Ry=90, Rz=90) 후 CAD 로컬 좌표계에서 상단 평면 중심 실측값:
#   X=-0.002, Y=-0.042, Z=0.031  (단위: m)
# T @ [x, y, z, 1] 로 월드 좌표 변환
CAD_PICK_LOCAL = np.array([0.000, -0.100, 0.031, 1.0])   # 상단 수평면 중심 (nz>0 삼각형 중심 실측값)

# 픽포인트 미세 조정 오프셋 (CAD 로컬 좌표계 기준, 단위: mm)
# 양수/음수로 픽포인트를 각 축 방향으로 이동
# X: 브라켓 폭 방향   (+: 오른쪽, -: 왼쪽)
# Y: 브라켓 길이 방향  (+: 앞,    -: 뒤)
# Z: 브라켓 높이 방향  (+: 위,    -: 아래)
PICK_OFFSET_X_MM = -5.0    # mm
PICK_OFFSET_Y_MM = 0.0    # mm
PICK_OFFSET_Z_MM = 0.0    # mm


# -----------------------------------------------------------------------------
# 회전 유틸
# -----------------------------------------------------------------------------

def _Rx(d: float) -> np.ndarray:
    a = np.radians(d)
    return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

def _Ry(d: float) -> np.ndarray:
    a = np.radians(d)
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])

def _Rz(d: float) -> np.ndarray:
    a = np.radians(d)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])


# -----------------------------------------------------------------------------
# CAD 모델 준비
# -----------------------------------------------------------------------------

def load_cad_as_pcd(stl_path: Path, n_points: int = CAD_SAMPLE_POINTS) -> o3d.geometry.PointCloud:
    """STL 로드 → mm→m 변환 → 축 보정 → 포인트 샘플링."""
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if not mesh.has_triangles():
        raise ValueError(f"STL 로드 실패: {stl_path}")

    # mm → m (원점 기준 — center=get_center()는 center 값이 변하지 않는 문제 있음)
    extent_before = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    STL 원본 extent: {np.round(extent_before, 2)} mm", flush=True)
    mesh.scale(1.0 / 1000.0, center=np.zeros(3))
    extent_after = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"    변환 후 extent:  {np.round(extent_after, 4)} m  "
          f"center={np.round(np.asarray(mesh.get_center()), 4)}", flush=True)

    # 축 보정 (CAD 자체 중심 기준 고정 회전)
    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix = np.eye(4)
    T_fix[:3, :3] = R
    T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)
    print(f"    축 보정: Rx={rx}° Ry={ry}° Rz={rz}°", flush=True)

    pcd = mesh.sample_points_poisson_disk(n_points)

    # 픽포인트 로컬 좌표 확인 출력
    pts = np.asarray(pcd.points)
    print(f"    CAD 픽포인트 로컬: "
          f"X={CAD_PICK_LOCAL[0]*1000:.1f}mm "
          f"Y={CAD_PICK_LOCAL[1]*1000:.1f}mm "
          f"Z={CAD_PICK_LOCAL[2]*1000:.1f}mm", flush=True)

    return pcd


# -----------------------------------------------------------------------------
# ICP (Point-to-Point, 다단계)
# -----------------------------------------------------------------------------

def run_icp_multistage(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Point-to-Point ICP를 단계적으로 수렴.

    coarse → fine 순으로 max_dist를 좁혀가며 수렴하므로
    초기 정렬 오차가 있어도 안정적으로 수렴.

    Returns:
        (T_final, fitness, rmse)
    """
    T = init_transform.copy()
    for stage in ICP_STAGES:
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            stage["max_dist"],
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=stage["max_iter"]
            ),
        )
        T = np.asarray(result.transformation)

    # 최종 fitness/rmse는 가장 촘촘한 거리 기준으로 재평가
    final = o3d.pipelines.registration.evaluate_registration(
        source, target, ICP_STAGES[-1]["max_dist"], T
    )
    return T, float(final.fitness), float(final.inlier_rmse)


# -----------------------------------------------------------------------------
# 뒤집힘 보정
# -----------------------------------------------------------------------------

def correct_flipped_pose(
    T: np.ndarray,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, float, float, bool]:
    """R[2,2] < 0 이면 Z축 180도 회전 후 ICP 재수렴.

    Returns:
        (T_final, fitness, rmse, was_flipped)
    """
    R = T[:3, :3]
    if R[2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            source, target, ICP_STAGES[-1]["max_dist"], T
        )
        return T, float(final.fitness), float(final.inlier_rmse), False

    # Z축 180도 회전 (translation 중심 고정)
    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4)
    T_flip[:3, :3] = R_flip
    c = T[:3, 3]
    T_flip[:3, 3] = c - R_flip @ c

    T_flipped = T_flip @ T
    T_final, fitness, rmse = run_icp_multistage(source, target, T_flipped)
    return T_final, fitness, rmse, True


# -----------------------------------------------------------------------------
# 변환행렬 → 6DoF
# -----------------------------------------------------------------------------

def transform_to_pose(T: np.ndarray) -> dict:
    """4x4 변환행렬 (m 단위) → xyz mm + ZYX 오일러각 deg."""
    xyz_mm = (T[:3, 3] * 1000.0).tolist()

    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])

    euler_deg = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm": [round(v, 3) for v in xyz_mm],
        "euler_deg": {
            "roll_deg":  round(euler_deg[0], 4),
            "pitch_deg": round(euler_deg[1], 4),
            "yaw_deg":   round(euler_deg[2], 4),
        },
        "transform_matrix": T.tolist(),
    }



# -----------------------------------------------------------------------------
# 픽포인트 계산
# -----------------------------------------------------------------------------

def compute_pick_point(T: np.ndarray) -> dict:
    """ICP 변환행렬로부터 픽포인트 위치와 접근 방향 계산.

    브라켓 상단 평면 = CAD 로컬 +Z 방향 (축 보정 후 확인됨).

    픽포인트 위치:
        CAD 중심에서 로컬 +Z 방향으로 CAD_HALF_Z_M 만큼 이동한 점.
        = T @ [0, 0, CAD_HALF_Z_M, 1]

    접근 방향(노말):
        변환행렬의 Z축 컬럼 = T[:3, 2]
        그리퍼가 이 방향의 반대(-approach)로 접근.

    Returns:
        {
            "position_mm": [x, y, z],     # 픽포인트 3D 위치 (mm)
            "approach_vec": [nx, ny, nz],  # 접근 방향 단위벡터 (면 노말)
            "approach_deg": {              # 접근 방향의 오일러각 표현
                "roll_deg", "pitch_deg", "yaw_deg"
            }
        }
    """
    # 픽포인트 위치: CAD 로컬 상단 평면 중심 + 오프셋 → 월드 좌표 변환
    pick_local = CAD_PICK_LOCAL.copy()
    pick_local[0] += PICK_OFFSET_X_MM / 1000.0
    pick_local[1] += PICK_OFFSET_Y_MM / 1000.0
    pick_local[2] += PICK_OFFSET_Z_MM / 1000.0
    world_top = T @ pick_local                         # (4,) homogeneous
    pick_pos_mm = (world_top[:3] * 1000.0).tolist()   # m → mm

    # 접근 방향: 변환 후 +Z 축 (면 노말)
    approach = T[:3, 2]                                # 단위벡터 (회전행렬 컬럼)
    approach = approach / (np.linalg.norm(approach) + 1e-9)

    # 접근 방향을 오일러각으로 표현 (ZYX)
    R_approach = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R_approach[2,0],
                  np.sqrt(R_approach[0,0]**2 + R_approach[1,0]**2))))
    cp = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R_approach[2,1]/cp, R_approach[2,2]/cp)))
        yaw  = float(np.degrees(np.arctan2(R_approach[1,0]/cp, R_approach[0,0]/cp)))
    else:
        roll = 0.0
        yaw  = float(np.degrees(np.arctan2(-R_approach[0,1], R_approach[1,1])))

    return {
        "position_mm": [round(v, 3) for v in pick_pos_mm],
        "approach_vec": [round(v, 6) for v in approach.tolist()],
        "approach_deg": {
            "roll_deg":  round(roll,  4),
            "pitch_deg": round(pitch, 4),
            "yaw_deg":   round(yaw,   4),
        },
    }


# -----------------------------------------------------------------------------
# 시각화
# -----------------------------------------------------------------------------

def save_icp_visualization(
    scene_pcd: o3d.geometry.PointCloud,
    cad_pcd: o3d.geometry.PointCloud,
    T: np.ndarray,
    pick: dict,
    out_path: Path,
) -> None:
    """scene(회색) + 변환된 CAD(초록) + 픽포인트(빨강 구) + 접근방향 화살표(파랑)를 PLY로 저장.

    Open3D에서 열었을 때:
        회색  = 실제 측정 포인트 클라우드 (scene)
        초록  = ICP 정합된 CAD 모델
        빨강  = 픽포인트 (그리퍼가 접촉할 위치)
        파랑  = 접근 방향 벡터 (면 노말 방향)
    """
    # scene (회색)
    scene_vis = copy.deepcopy(scene_pcd)
    n = len(np.asarray(scene_vis.points))
    scene_vis.colors = o3d.utility.Vector3dVector(np.tile([0.6, 0.6, 0.6], (n, 1)))

    # CAD 정합 결과 (초록)
    cad_vis = copy.deepcopy(cad_pcd)
    cad_vis.transform(T)
    n = len(np.asarray(cad_vis.points))
    cad_vis.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.9, 0.3], (n, 1)))

    # 픽포인트 구 (빨강) — 반지름 5mm
    pick_pos_m = np.array(pick["position_mm"]) / 1000.0
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.translate(pick_pos_m)
    sphere.paint_uniform_color([1.0, 0.1, 0.1])
    sphere_pcd = sphere.sample_points_uniformly(number_of_points=500)

    # 접근 방향 화살표 (파랑) — 픽포인트에서 30mm 길이
    approach = np.array(pick["approach_vec"])
    arrow_end = pick_pos_m + approach * 0.03   # 30mm
    n_arrow = 50
    t_vals = np.linspace(0, 1, n_arrow)
    arrow_pts = np.array([pick_pos_m + t * (arrow_end - pick_pos_m) for t in t_vals])
    arrow_pcd = o3d.geometry.PointCloud()
    arrow_pcd.points = o3d.utility.Vector3dVector(arrow_pts)
    arrow_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (n_arrow, 1)))

    combined = scene_vis + cad_vis + sphere_pcd + arrow_pcd
    o3d.io.write_point_cloud(str(out_path), combined, write_ascii=False)


# -----------------------------------------------------------------------------
# 단일 인스턴스 처리
# -----------------------------------------------------------------------------

def process_instance(
    instance_ply: Path,
    cad_pcd: o3d.geometry.PointCloud,
    cad_down: o3d.geometry.PointCloud,
    output_dir: Path,
) -> dict:
    """중심정렬 + 다단계 ICP로 6DoF 자세 추정.

    FPFH/RANSAC 없음 → 브라켓이 항상 비슷한 방향으로 놓인 경우에 적합.
    """
    stem = instance_ply.stem

    # scene 로드
    scene_pcd = o3d.io.read_point_cloud(str(instance_ply))
    n_pts = len(np.asarray(scene_pcd.points))
    if n_pts < 50:
        return {"file": stem, "error": f"포인트 부족: {n_pts}개"}

    print(f"  {stem}: {n_pts} pts", flush=True)

    # scene 전처리 (노이즈 제거 + 다운샘플 — 노말 계산 없음)
    n_before = n_pts
    scene_clean, _ = scene_pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO,
    )
    n_after = len(np.asarray(scene_clean.points))
    scene_down = scene_clean.voxel_down_sample(VOXEL_SIZE_SCENE)
    n_scene_down = len(np.asarray(scene_down.points))
    n_cad_down   = len(np.asarray(cad_down.points))

    removal_pct = (1 - n_after / max(n_before, 1)) * 100
    print(f"    노이즈 제거: {n_before} → {n_after} pts ({removal_pct:.1f}% 제거)", flush=True)
    print(f"    다운샘플:    scene={n_scene_down}  cad={n_cad_down}", flush=True)

    # 중심 정렬 초기화
    src_c = np.asarray(cad_down.get_center())
    tgt_c = np.asarray(scene_down.get_center())
    T_init = np.eye(4)
    T_init[:3, 3] = tgt_c - src_c
    print(f"    중심 정렬:   {np.round(src_c,3)} → {np.round(tgt_c,3)}", flush=True)

    # 다단계 ICP
    print(f"    ICP 중...", flush=True)
    T_final, fitness, rmse = run_icp_multistage(cad_down, scene_down, T_init)
    print(f"    ICP fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)

    # 뒤집힘 보정
    T_final, fitness, rmse, was_flipped = correct_flipped_pose(
        T_final, cad_down, scene_down
    )
    if was_flipped:
        print(f"    △ 뒤집힘 감지 → Z축 180도 보정 후 재수렴", flush=True)
        print(f"    보정 후 fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)

    t_mm = np.round(T_final[:3, 3] * 1000, 1)
    print(f"    T translation: {t_mm} mm", flush=True)

    # 실패 판정
    if fitness < ICP_FITNESS_THRESHOLD:
        print(f"    ✗ ICP 실패 (fitness={fitness:.4f} < {ICP_FITNESS_THRESHOLD})", flush=True)
        return {"file": stem, "error": "ICP 정합 실패",
                "icp_fitness": float(fitness), "icp_rmse": float(rmse)}

    if max(abs(v) for v in T_final[:3, 3]) > XYZ_MAX_M:
        print(f"    ✗ xyz 비정상: {t_mm} mm → 로컬 미니멈", flush=True)
        return {"file": stem, "error": "xyz 범위 이상",
                "icp_fitness": float(fitness)}

    pose = transform_to_pose(T_final)
    xyz  = pose["xyz_mm"]
    eul  = pose["euler_deg"]

    # 픽포인트 계산
    pick = compute_pick_point(T_final)
    ppos = pick["position_mm"]
    avec = pick["approach_vec"]
    adeg = pick["approach_deg"]

    # 저장
    vis_path = output_dir / f"{stem}_icp_vis.ply"
    save_icp_visualization(scene_pcd, cad_pcd, T_final, pick, vis_path)

    result = {
        "file": stem,
        "input_ply": str(instance_ply.relative_to(ROOT)),
        "num_points_scene": n_pts,
        "num_points_after_outlier_removal": n_after,
        "icp_fitness": float(fitness),
        "icp_rmse_m": float(rmse),
        "was_flipped": was_flipped,
        "pose": pose,
        "pick_point": pick,
        "vis_ply": str(vis_path.relative_to(ROOT)),
    }
    with (output_dir / f"{stem}_pose.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"    ✓ CAD 중심:   xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm  "
          f"roll={eul['roll_deg']:.1f}° pitch={eul['pitch_deg']:.1f}° "
          f"yaw={eul['yaw_deg']:.1f}°", flush=True)
    print(f"    ✓ 픽포인트:   xyz=({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm", flush=True)
    print(f"    ✓ 접근 방향:  vec=({avec[0]:.3f}, {avec[1]:.3f}, {avec[2]:.3f})  "
          f"roll={adeg['roll_deg']:.1f}° pitch={adeg['pitch_deg']:.1f}° "
          f"yaw={adeg['yaw_deg']:.1f}°", flush=True)

    return result


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------

def main() -> int:
    for path, name in [(CAD_PATH, "CAD"), (INFERENCE_DIR, "추론 결과 폴더")]:
        if not path.exists():
            print(f"ERROR: {name} 없음: {path}", flush=True)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  Stage 5-B: ICP 정합 → 6DoF 자세 추정 (경량 버전)", flush=True)
    print("=" * 70, flush=True)
    print(f"  CAD:         {CAD_PATH}", flush=True)
    print(f"  축 보정:     Rx={CAD_AXIS_CORRECTION_DEG[0]}° "
          f"Ry={CAD_AXIS_CORRECTION_DEG[1]}° Rz={CAD_AXIS_CORRECTION_DEG[2]}°", flush=True)
    print(f"  Inference:   {INFERENCE_DIR}", flush=True)
    print(f"  Voxel CAD:   {VOXEL_SIZE_CAD*1000:.1f} mm", flush=True)
    print(f"  Voxel scene: {VOXEL_SIZE_SCENE*1000:.1f} mm", flush=True)
    stages_str = " → ".join(f"{s['max_dist']*1000:.0f}mm×{s['max_iter']}" for s in ICP_STAGES)
    print(f"  ICP 단계:    {stages_str}", flush=True)
    print(f"  픽 오프셋:   X={PICK_OFFSET_X_MM:+.1f}mm  "
          f"Y={PICK_OFFSET_Y_MM:+.1f}mm  Z={PICK_OFFSET_Z_MM:+.1f}mm", flush=True)
    print("=" * 70, flush=True)

    # CAD 로드 + 다운샘플 (1회)
    print("\n[1] CAD 모델 로드 중...", flush=True)
    try:
        cad_pcd = load_cad_as_pcd(CAD_PATH)
    except Exception as e:
        print(f"ERROR: CAD 로드 실패: {e}", flush=True)
        return 1
    cad_down = cad_pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    print(f"    ✓ CAD 샘플: {len(np.asarray(cad_pcd.points))}pts  "
          f"다운샘플: {len(np.asarray(cad_down.points))}pts "
          f"(voxel={VOXEL_SIZE_CAD*1000:.1f}mm)", flush=True)
    print(f"    CAD center: {np.round(np.asarray(cad_down.get_center()), 4)} m", flush=True)

    # 인스턴스 PLY 목록
    instance_plys = sorted(INFERENCE_DIR.glob("frame_*_obj*.ply"))
    instance_plys = [
        p for p in instance_plys
        if "_colored" not in p.stem and "_icp_vis" not in p.stem
    ]
    if not instance_plys:
        print("\nERROR: 인스턴스 PLY 없음. Stage 5-A를 먼저 실행하세요.", flush=True)
        return 1

    print(f"\n[2] ICP 정합: {len(instance_plys)}개 인스턴스", flush=True)
    print("-" * 70, flush=True)

    all_results = []
    n_success, n_fail = 0, 0

    for ply_path in instance_plys:
        result = process_instance(ply_path, cad_pcd, cad_down, OUTPUT_DIR)
        all_results.append(result)
        if "error" in result:
            n_fail += 1
        else:
            n_success += 1

    with (OUTPUT_DIR / "icp_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "input_data": input_data,
            "cad_path": str(CAD_PATH),
            "cad_axis_correction_deg": list(CAD_AXIS_CORRECTION_DEG),
            "voxel_size_cad_m": VOXEL_SIZE_CAD,
            "voxel_size_scene_m": VOXEL_SIZE_SCENE,
            "icp_stages": ICP_STAGES,
            "icp_fitness_threshold": ICP_FITNESS_THRESHOLD,
            "total": len(instance_plys),
            "success": n_success,
            "failed": n_fail,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print("-" * 70, flush=True)
    print(f"\n[3] 요약", flush=True)
    print(f"  전체 인스턴스:   {len(instance_plys)}", flush=True)
    print(f"  정합 성공:       {n_success}", flush=True)
    print(f"  정합 실패:       {n_fail}", flush=True)
    print(f"\n  결과 위치: {OUTPUT_DIR}", flush=True)
    print(f"  - *_pose.json:      6DoF 자세", flush=True)
    print(f"  - *_icp_vis.ply:    정합 시각화 (회색=scene, 초록=CAD)", flush=True)
    print(f"\n  ✓ Stage 5-B 완료", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())