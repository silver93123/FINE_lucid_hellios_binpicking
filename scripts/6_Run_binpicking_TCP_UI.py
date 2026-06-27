"""실전 빈피킹 파이프라인: TCP 서버 모드

흐름:
  [1] RTMDet 모델 로드
  [2] CAD 모델 로드
  [3] 카메라 초기화 + 워밍업
  [4] TCP 서버 시작 → 클라이언트 연결 대기
  [5] "CAPTURE" 수신 → 캡처 → Detection → ICP → 픽포인트 JSON 응답
       "QUIT" 수신 → 서버 종료

실행:
  cd ~/FINE_RTMDet
  python scripts/6_Run_binpicking_TCP_UI.py

옵션:
  --config  config/config.yaml   카메라 설정 파일 (기본값)
  --warmup  3                    워밍업 프레임 수
  --out     data/captures/live   캡처 저장 경로
  --host    192.168.0.22         TCP 바인드 주소
  --port    29999                TCP 포트

TCP 프로토콜:
  클라이언트 → 서버:
    "CAPTURE\\n"   캡처 + 픽포인트 계산 요청
    "QUIT\\n"      서버 종료
  서버 → 클라이언트 (문자열 + 개행):
    {'ok', 3, (x, y, z, roll, pitch, yaw, fit), (...), (...)}
    {'No'}
    {'error', 'message'}

변경 이력:
  - [패치] ICP 초기값: 단순 평행이동 → 고정 자세(ICP_INIT_*_DEG) 기반 초기 회전 추가
  - [패치] ICP 결과 회전 구속 검사: roll/pitch 허용 범위 초과 시 기각
           ICP_ROLL_RANGE, ICP_PITCH_RANGE, ICP_YAW_RANGE 상수로 조정
"""

from __future__ import annotations

import argparse
import copy
import json
import socket
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

import logging

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d 필요. pip install open3d", flush=True)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera import create_camera       # noqa: E402
from src.detection import RTMDetInferencer  # noqa: E402

# =============================================================================
# 설정
# =============================================================================

# ── Detection ─────────────────────────────────────────────────────────────────
WORK_DIR      = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1"
CONFIG_PATH   = WORK_DIR / "rtmdet-ins_bracket.py"

_candidates = sorted(WORK_DIR.glob("best_*.pth"))
if not _candidates:
    print(f"ERROR: best 모델이 없습니다: {WORK_DIR}", flush=True)
    sys.exit(1)
CHECKPOINT_PATH = _candidates[-1]

SCORE_THRESHOLD        = 0.3
MIN_POINTS_PER_INSTANCE = 100
MASK_IOU_THRESHOLD     = 0.6

# ── ICP ───────────────────────────────────────────────────────────────────────
CAD_PATH           = ROOT / "data" / "cad" / "bracket_v2.stl"
CAD_SAMPLE_POINTS  = 20000
VOXEL_SIZE_CAD     = 0.002
VOXEL_SIZE_SCENE   = 0.003
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

ICP_STAGES = [
    {"max_dist": 0.020, "max_iter": 100},
    {"max_dist": 0.010, "max_iter": 100},
    {"max_dist": 0.005, "max_iter": 100},
]

ICP_FITNESS_THRESHOLD = 0.5
XYZ_MAX_M             = 2.0
CAD_AXIS_CORRECTION_DEG = (-90, 90, 90)

# ── ICP 고정 초기 자세 ────────────────────────────────────────────────
# 브라켓이 항상 특정 자세에서 크게 벗어나지 않을 때 사용합니다.
# 정상 케이스 로그(roll/pitch/yaw)를 수 회 관찰한 평균값을 입력하세요.
# 적용 순서: Rz(yaw) @ Ry(pitch) @ Rx(roll)
ICP_INIT_ROLL_DEG  =  0.0   # [deg] 브라켓 초기 roll  (관찰 평균값으로 교체)
ICP_INIT_PITCH_DEG =  0.0   # [deg] 브라켓 초기 pitch (관찰 평균값으로 교체)
ICP_INIT_YAW_DEG   =  0.0   # [deg] 브라켓 초기 yaw   (관찰 평균값으로 교체)

# ── ICP 회전 구속 조건 ─────────────────────────────────────────────────
# ICP 결과 roll/pitch/yaw 가 허용 범위를 벗어나면 기각합니다.
# 정상 케이스 로그를 수 회 관찰한 뒤 최댓값 + 여유 ±10° 로 설정하세요.
ICP_ROLL_RANGE  = (-30.0,  30.0)   # [deg] 허용 roll  범위
ICP_PITCH_RANGE = (-30.0,  30.0)   # [deg] 허용 pitch 범위
ICP_YAW_RANGE   = (-180.0, 180.0)  # [deg] yaw 는 360° 자유 (필요 시 좁히세요)

# ── 픽포인트 ──────────────────────────────────────────────────────────────────
CAD_PICK_LOCAL   = np.array([0.000, -0.100, 0.031, 1.0])
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
# 로거
# =============================================================================
_logger: logging.Logger | None = None

def log(msg: str) -> None:
    print(msg, flush=True)
    if _logger is not None:
        _logger.info(msg)

def setup_file_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "binpicking.log"
    logger = logging.getLogger("binpicking")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sep        = "=" * 70
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n{sep}")
    logger.info(f" 서버 기동: {start_time}")
    logger.info(sep)
    return logger

# =============================================================================
# 디렉터리 / 캡처 저장
# =============================================================================
def setup_dirs(out_dir: Path) -> dict[str, Path]:
    subdirs = {
        "intensity":              out_dir / "intensity",
        "pointcloud_organized":   out_dir / "pointcloud_organized",
        "valid_mask":             out_dir / "valid_mask",
        "metadata":               out_dir / "metadata",
        "results":                out_dir / "results",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs

def save_capture(frame, dirs: dict, idx: int, cfg_camera: dict) -> dict:
    dt_name        = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_intensity = f"intensity_{dt_name}.png"
    fname_pcd       = f"pointcloud_{dt_name}.npy"
    fname_mask      = f"mask_{dt_name}.npy"
    fname_meta      = f"metadata_{dt_name}.json"

    cv2.imwrite(str(dirs["intensity"] / fname_intensity), frame.intensity)
    np.save(dirs["pointcloud_organized"] / fname_pcd,
            frame.points_organized.astype(np.float32))
    np.save(dirs["valid_mask"] / fname_mask,
            frame.valid_mask.astype(bool))

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
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "image":  {"width": int(frame.width), "height": int(frame.height)},
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
            "intensity":            f"intensity/{fname_intensity}",
            "pointcloud_organized": f"pointcloud_organized/{fname_pcd}",
            "valid_mask":           f"valid_mask/{fname_mask}",
        },
    }
    with (dirs["metadata"] / fname_meta).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata

# =============================================================================
# Detection + PCD 분리
# =============================================================================
def overlay_results(image_bgr, results, valid_mask=None):
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

def draw_picks_on_overlay(image_bgr: np.ndarray, picks_2d: list) -> np.ndarray:
    out   = image_bgr.copy()
    H, W  = out.shape[:2]
    for i, (px, py, pick, icp_fitness, bbox) in enumerate(picks_2d):
        color        = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
        pp           = pick["position_mm"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(px), int(py)), color,
                       cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        line1 = f"#{i} ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm"
        line2 = f"ICP fit: {icp_fitness:.3f}"
        font, font_scale, thickness, line_gap = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1, 4
        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        box_w = max(w1, w2) + 8
        box_h = h1 + h2 + line_gap + 8
        tx = max(x1, 0)
        ty = y1 - box_h - 4
        if ty < 0:
            ty = y2 + 4
        ty = min(ty, H - box_h - 2)
        tx = min(tx, W - box_w - 2)
        cv2.rectangle(out, (tx - 2, ty), (tx + box_w, ty + box_h), (0, 0, 0), -1)
        cv2.putText(out, line1, (tx + 2, ty + h1 + 2),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(out, line2, (tx + 2, ty + h1 + h2 + line_gap + 4),
                    font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
    return out

# =============================================================================
# 실시간 모니터 창
# =============================================================================
_WIN = "BinPicking Monitor"

def _put(img, text, x, y, color=(220, 220, 220), scale=0.46, thickness=1):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def build_info_panel(h: int, w: int, info: dict) -> np.ndarray:
    MIN_PANEL_H = 680
    ph    = max(h, MIN_PANEL_H)
    panel = np.full((ph, w, 3), 28, dtype=np.uint8)
    x0, lh = 12, 20

    def sep(y, c=(55, 55, 55)):
        cv2.line(panel, (x0, y), (w - x0, y), c, 1)

    def section(text, y, color=(100, 200, 255)):
        _put(panel, text, x0, y, color=color, scale=0.47)

    def row(text, y, color=(190, 190, 190)):
        _put(panel, text, x0 + 6, y, color=color, scale=0.41)

    y = 19
    _put(panel, "BinPicking Monitor", x0, y, color=(255, 200, 60), scale=0.53, thickness=1)
    y += lh + 2; sep(y); y += lh

    status  = info.get("status", "Waiting")
    s_color = {"Waiting": (150, 150, 150), "Processing": (60, 200, 255),
               "Done": (60, 220, 60), "No detection": (60, 140, 255),
               "Error": (60, 60, 220)}.get(status, (200, 200, 200))

    section("[ Server Status ]", y); y += lh
    row(f"Status : {status}", y, color=s_color);          y += lh
    row(f"Frame # : {info.get('frame_idx', '-')}", y);    y += lh
    row(f"Time : {info.get('timestamp', '-')}", y);       y += lh + 2
    sep(y); y += lh

    section("[ Capture Info ]", y); y += lh
    row(f"Capture : {info.get('capture_ms', 0):.0f} ms",  y); y += lh
    row(f"Valid : {info.get('valid_ratio', 0):.1f} %",    y); y += lh
    row(f"Z range : {info.get('z_min', 0):.0f} ~ {info.get('z_max', 0):.0f} mm", y)
    y += lh + 2; sep(y); y += lh

    section("[ Timing ]", y); y += lh
    row(f"Detection: {info.get('det_ms', 0):.0f} ms", y); y += lh
    row(f"ICP : {info.get('icp_ms', 0):.0f} ms",      y); y += lh
    total = info.get("capture_ms", 0) + info.get("det_ms", 0) + info.get("icp_ms", 0)
    row(f"Total : {total:.0f} ms", y, color=(255, 200, 60)); y += lh + 2
    sep(y); y += lh

    picks = info.get("picks", [])
    section(f"[ Results : {len(picks)} obj(s) ]", y); y += lh
    if not picks:
        row(" No objects detected", y, color=(100, 100, 220))
    else:
        for i, pk in enumerate(picks):
            pp  = pk["position_mm"]
            deg = pk["approach_deg"]
            fit = pk["icp_fitness"]
            ci  = tuple(int(v) for v in _PALETTE_BGR[i % len(_PALETTE_BGR)])
            sep(y, c=(45, 45, 45)); y += lh - 4
            _put(panel, f" Obj #{i}", x0, y, color=ci, scale=0.43)
            y += lh - 2
            row(f" X= {pp[0]:+8.2f} Y= {pp[1]:+8.2f}", y);     y += lh - 3
            row(f" Z= {pp[2]:+8.2f} mm", y);                     y += lh - 3
            row(f" R= {deg['roll_deg']:+7.2f} P= {deg['pitch_deg']:+7.2f}", y); y += lh - 3
            row(f" Yaw= {deg['yaw_deg']:+7.2f} deg", y);         y += lh - 3
            fc = (60, 220, 60) if fit >= 0.7 else (60, 140, 255) if fit >= 0.5 else (60, 60, 220)
            row(f" ICP fit : {fit:.3f}", y, color=fc);           y += lh

    sep(ph - 22)
    _put(panel, "QUIT=exit  ESC=close window",
         x0, ph - 8, color=(80, 80, 80), scale=0.37)
    return panel

def show_monitor(overlay_bgr: np.ndarray, info: dict):
    PANEL_W = 310
    SCALE   = 1.5
    oh, ow  = overlay_bgr.shape[:2]
    overlay_bgr = cv2.resize(overlay_bgr,
                              (int(ow * SCALE), int(oh * SCALE)),
                              interpolation=cv2.INTER_LINEAR)
    img_h = overlay_bgr.shape[0]
    panel = build_info_panel(img_h, PANEL_W, info)
    ph    = panel.shape[0]
    if img_h < ph:
        pad = np.zeros((ph - img_h, overlay_bgr.shape[1], 3), dtype=np.uint8)
        overlay_bgr = np.vstack([overlay_bgr, pad])
    divider = np.full((ph, 2, 3), 70, dtype=np.uint8)
    canvas  = np.hstack([overlay_bgr, divider, panel])
    cv2.imshow(_WIN, canvas)
    cv2.waitKey(1)

def init_monitor(h: int, w: int):
    blank = np.full((h, w, 3), 18, dtype=np.uint8)
    _put(blank, "Waiting for capture command...",
         w // 2 - 160, h // 2, color=(110, 110, 110), scale=0.58)
    show_monitor(blank, {"status": "Waiting", "frame_idx": 0,
                         "timestamp": "-", "picks": []})

# =============================================================================
# 마스크 NMS
# =============================================================================
def mask_nms(results, iou_threshold: float = MASK_IOU_THRESHOLD):
    keep, removed  = [], []
    suppressed     = [False] * len(results)
    for i, ri in enumerate(results):
        if suppressed[i]:
            continue
        keep.append(ri)
        area_i = ri.mask.sum()
        if area_i == 0:
            continue
        for j in range(i + 1, len(results)):
            if suppressed[j]:
                continue
            rj    = results[j]
            inter = (ri.mask & rj.mask).sum()
            if inter == 0:
                continue
            area_j = rj.mask.sum()
            union  = area_i + area_j - inter
            iou    = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                suppressed[j] = True
                removed.append((rj, ri, float(iou)))
    return keep, removed

# =============================================================================
# PCD 저장 헬퍼
# =============================================================================
def save_instance_pcd(points, out_path, color):
    if points.size == 0:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64), (len(points), 1)))
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))

def save_colored_full_pcd(pcd_organized, valid_mask, results, out_path):
    all_pts = pcd_organized[valid_mask]
    if len(all_pts) == 0:
        return False
    colors = np.tile(_BG_COLOR, (len(all_pts), 1))
    H, W   = valid_mask.shape
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

# =============================================================================
# Detection 실행
# =============================================================================
def run_detection(frame_name, gray, pcd_organized, valid_mask, inferencer, result_dir):
    H, W = gray.shape
    bgr  = np.stack([gray, gray, gray], axis=-1)

    results = inferencer.infer(bgr)
    results, nms_removed = mask_nms(results)
    if nms_removed:
        for rem, winner, iou in nms_removed:
            print(f" [NMS] score={rem.score:.2f} 제거 "
                  f"(IoU={iou:.2f}, winner score={winner.score:.2f})", flush=True)

    cv2.imwrite(str(result_dir / f"{frame_name}_overlay.png"),
                overlay_results(bgr, results, valid_mask))
    save_colored_full_pcd(pcd_organized, valid_mask, results,
                          result_dir / f"{frame_name}_colored.ply")

    instances_info, instance_plys = [], []
    for i, r in enumerate(results):
        combined = r.mask & valid_mask
        obj_pts  = pcd_organized[combined]
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
        cx_2d     = float((r.bbox[0] + r.bbox[2]) / 2)
        cy_2d     = float((r.bbox[1] + r.bbox[3]) / 2)
        instances_info.append({
            "instance_id": i, "class": r.class_name,
            "score": float(r.score),
            "num_points_3d": int(len(obj_pts)),
            "center_mm": center.tolist(), "size_mm": size.tolist(),
            "bbox_center_2d": [cx_2d, cy_2d],
        })
        if ok:
            instance_plys.append((ply_path, cx_2d, cy_2d, r.bbox))

    summary = {
        "frame": frame_name,
        "num_detected":  len(results),
        "num_with_pcd":  len(instance_plys),
        "instances":     instances_info,
    }

    instance_mask_union = np.zeros(valid_mask.shape, dtype=bool)
    for r in results:
        instance_mask_union |= (r.mask & valid_mask)
    bg_only_mask = valid_mask & ~instance_mask_union
    bg_pts       = pcd_organized[bg_only_mask]
    bg_pcd       = o3d.geometry.PointCloud()
    if len(bg_pts) > 0:
        bg_pcd.points = o3d.utility.Vector3dVector(bg_pts / 1000.0)
        bg_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.55, 0.55, 0.55], (len(bg_pts), 1)))

    return summary, instance_plys, bgr, bg_pcd

# =============================================================================
# ICP 관련 유틸
# =============================================================================
def _Rx(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
    return R

def _Ry(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c
    return R

def _Rz(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R

def load_cad_as_pcd(cad_path):
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    ext  = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    if ext.max() > 10.0:
        mesh.scale(1.0 / 1000.0, center=np.zeros(3))
    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R      = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix  = np.eye(4); T_fix[:3, :3] = R; T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)
    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)

# ── [패치] 고정 자세 기반 ICP 초기값 ────────────────────────────────────────
def build_icp_init(scene_down, cad_down) -> np.ndarray:
    """고정 초기 자세(ICP_INIT_*_DEG) 기반 T_init 생성.

    브라켓이 항상 비슷한 자세로 놓이는 경우, 관찰된 평균 자세를
    고정값으로 지정하는 방식이 안정적입니다.

    설정 방법:
      1. ICP_ROLL/PITCH/YAW_RANGE 를 일시적으로 (-180, 180) 으로 열어두고
         정상 케이스 10회 실행
      2. 로그에 출력되는 roll/pitch/yaw 평균값 확인
      3. ICP_INIT_ROLL_DEG / PITCH_DEG / YAW_DEG 에 입력 후 범위 다시 조임

    초기 회전 적용 순서: Rz(yaw) @ Ry(pitch) @ Rx(roll)
    평행이동: CAD 중심 → 씬 중심 으로 이동
    """
    R_init = _Rz(ICP_INIT_YAW_DEG) @ _Ry(ICP_INIT_PITCH_DEG) @ _Rx(ICP_INIT_ROLL_DEG)

    sc_center = np.asarray(scene_down.get_center())
    cd_center = np.asarray(cad_down.get_center())

    T_init = np.eye(4)
    T_init[:3, :3] = R_init
    T_init[:3, 3]  = sc_center - R_init @ cd_center
    return T_init


# ── [패치] ICP 결과 회전 구속 검사 ──────────────────────────────────────────
def check_rotation_constraint(T: np.ndarray):
    """ICP 결과 행렬의 roll/pitch/yaw 가 허용 범위 내인지 검사.

    Args:
        T: 4×4 변환 행렬 (ICP 결과)

    Returns:
        ok  (bool): True = 허용 범위 내, False = 범위 초과(기각 대상)
        msg (str) : 상태 메시지 (로그 출력용)
    """
    R     = T[:3, :3]
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2)))
    cp    = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp))
        yaw  = np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp))
    else:
        roll, yaw = 0.0, np.degrees(np.arctan2(-R[0, 1], R[1, 1]))

    violations = []
    if not (ICP_ROLL_RANGE[0]  <= roll  <= ICP_ROLL_RANGE[1]):
        violations.append(
            f"roll={roll:.1f}° (허용 [{ICP_ROLL_RANGE[0]}, {ICP_ROLL_RANGE[1]}])")
    if not (ICP_PITCH_RANGE[0] <= pitch <= ICP_PITCH_RANGE[1]):
        violations.append(
            f"pitch={pitch:.1f}° (허용 [{ICP_PITCH_RANGE[0]}, {ICP_PITCH_RANGE[1]}])")
    if not (ICP_YAW_RANGE[0]   <= yaw   <= ICP_YAW_RANGE[1]):
        violations.append(
            f"yaw={yaw:.1f}° (허용 [{ICP_YAW_RANGE[0]}, {ICP_YAW_RANGE[1]}])")

    if violations:
        return False, "회전 구속 위반: " + ", ".join(violations)
    return True, f"roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°"


# ── 기존 ICP 함수 (변경 없음) ────────────────────────────────────────────────
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
    T_flip = np.eye(4); T_flip[:3, :3] = R_flip
    c = T[:3, 3]; T_flip[:3, 3] = c - R_flip @ c
    T_f, fit, rmse = run_icp_multistage(src, tgt, T_flip @ T)
    return T_f, fit, rmse, True

def transform_to_pose(T):
    xyz_mm = (T[:3, 3] * 1000.0).tolist()
    R      = T[:3, :3]
    pitch  = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    cp     = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    e = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm":    [round(v, 3) for v in xyz_mm],
        "euler_deg": {"roll_deg":  round(e[0], 4),
                      "pitch_deg": round(e[1], 4),
                      "yaw_deg":   round(e[2], 4)},
        "transform_matrix": T.tolist(),
    }

def compute_pick_point(T):
    pl = CAD_PICK_LOCAL.copy()
    pl[0] += PICK_OFFSET_X_MM / 1000.0
    pl[1] += PICK_OFFSET_Y_MM / 1000.0
    pl[2] += PICK_OFFSET_Z_MM / 1000.0
    wt  = T @ pl
    pos = (wt[:3] * 1000.0).tolist()
    R   = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))))
    cp    = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp)))
        yaw  = float(np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp)))
    else:
        roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
    return {
        "position_mm":  [round(v, 3) for v in pos],
        "approach_deg": {"roll_deg":  round(roll,  4),
                         "pitch_deg": round(pitch, 4),
                         "yaw_deg":   round(yaw,   4)},
    }

def build_icp_elements(scene_pcd, cad_pcd, T, pick, inst_color):
    sv = copy.deepcopy(scene_pcd)
    sv.colors = o3d.utility.Vector3dVector(
        np.tile(inst_color, (len(np.asarray(sv.points)), 1)))
    cv = copy.deepcopy(cad_pcd); cv.transform(T)
    cv.colors = o3d.utility.Vector3dVector(
        np.tile([0.1, 0.9, 0.3], (len(np.asarray(cv.points)), 1)))
    pm = np.array(pick["position_mm"]) / 1000.0
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sp.translate(pm); sp.paint_uniform_color([1.0, 0.1, 0.1])
    sp_pcd = sp.sample_points_uniformly(500)
    deg = pick["approach_deg"]
    cr, sr = np.cos(np.radians(deg["roll_deg"])),  np.sin(np.radians(deg["roll_deg"]))
    cp_, sp_ = np.cos(np.radians(deg["pitch_deg"])), np.sin(np.radians(deg["pitch_deg"]))
    cy, sy = np.cos(np.radians(deg["yaw_deg"])),  np.sin(np.radians(deg["yaw_deg"]))
    app = np.array([cr * sy * sp_ + sr * cy,
                    sr * sy - cr * cy * sp_,
                    cr * cp_])
    app = app / (np.linalg.norm(app) + 1e-9)
    ap  = np.array([pm + t * app * 0.03 for t in np.linspace(0, 1, 50)])
    ap_pcd = o3d.geometry.PointCloud()
    ap_pcd.points = o3d.utility.Vector3dVector(ap)
    ap_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (50, 1)))
    return sv + cv + sp_pcd + ap_pcd

# =============================================================================
# ICP 프레임 처리  ★ 핵심 변경 부분
# =============================================================================
def run_icp_for_frame(instance_plys, cad_pcd, cad_down,
                      result_dir, frame_name, bgr_image, bg_pcd=None):
    icp_results  = []
    picks_2d     = []
    combined_pcd = bg_pcd if bg_pcd is not None else o3d.geometry.PointCloud()

    for ply_path, cx_2d, cy_2d, bbox in instance_plys:
        stem     = ply_path.stem
        inst_idx = int(stem.split("obj")[-1])

        scene_pcd = o3d.io.read_point_cloud(str(ply_path))
        n_pts     = len(np.asarray(scene_pcd.points))
        if n_pts < 50:
            icp_results.append({"instance_id": inst_idx,
                                 "error": f"포인트 부족: {n_pts}개"})
            continue

        log(f"   obj{inst_idx}: {n_pts} pts")
        sc, _   = scene_pcd.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS,
                                                        OUTLIER_STD_RATIO)
        n_after = len(np.asarray(sc.points))
        sd      = sc.voxel_down_sample(VOXEL_SIZE_SCENE)

        # ── [패치] 고정 자세 기반 초기값 ────────────────────────────────
        T_init = build_icp_init(sd, cad_down)
        log(f"   T_init roll={ICP_INIT_ROLL_DEG}° pitch={ICP_INIT_PITCH_DEG}° "
            f"yaw={ICP_INIT_YAW_DEG}° "
            f"t=[{T_init[0,3]:.4f}, {T_init[1,3]:.4f}, {T_init[2,3]:.4f}]")

        T, fit, rmse = run_icp_multistage(cad_down, sd, T_init)
        T, fit, rmse, flipped = correct_flipped_pose(T, cad_down, sd)
        if flipped:
            log(f"   △ 뒤집힘 보정 후 fitness={fit:.4f}")

        # ── fitness 검사 ─────────────────────────────────────────────────
        if fit < ICP_FITNESS_THRESHOLD:
            log(f"   ✗ ICP 실패 (fitness={fit:.4f})")
            icp_results.append({"instance_id": inst_idx,
                                 "error": "ICP 정합 실패",
                                 "icp_fitness": float(fit)})
            ply_path.unlink(missing_ok=True)
            continue

        # ── xyz 범위 검사 ────────────────────────────────────────────────
        if max(abs(v) for v in T[:3, 3]) > XYZ_MAX_M:
            icp_results.append({"instance_id": inst_idx,
                                 "error": "xyz 범위 이상",
                                 "icp_fitness": float(fit)})
            ply_path.unlink(missing_ok=True)
            continue

        # ── [패치] 회전 구속 검사 ────────────────────────────────────────
        rot_ok, rot_msg = check_rotation_constraint(T)
        if not rot_ok:
            log(f"   ✗ {rot_msg} → 기각")
            icp_results.append({"instance_id": inst_idx,
                                 "error": rot_msg,
                                 "icp_fitness": float(fit)})
            ply_path.unlink(missing_ok=True)
            continue
        log(f"   ✓ 회전 OK: {rot_msg}")

        # ── 정상 결과 처리 ───────────────────────────────────────────────
        pose = transform_to_pose(T)
        pick = compute_pick_point(T)
        ppos = pick["position_mm"]
        deg  = pick["approach_deg"]

        inst_color   = _PALETTE_RGB_FLOAT[inst_idx % len(_PALETTE_RGB_FLOAT)].tolist()
        combined_pcd += build_icp_elements(scene_pcd, cad_pcd, T, pick, inst_color)
        picks_2d.append((cx_2d, cy_2d, pick, float(fit), bbox))

        result = {
            "instance_id":  inst_idx,
            "icp_fitness":  float(fit),
            "icp_rmse_m":   float(rmse),
            "was_flipped":  flipped,
            "num_points_scene":                 n_pts,
            "num_points_after_outlier_removal": n_after,
            "pose":       pose,
            "pick_point": pick,
        }
        print(f"   ✓ 픽포인트: ({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm "
              f"fit={fit:.3f} roll={deg['roll_deg']:.2f} "
              f"pitch={deg['pitch_deg']:.2f} yaw={deg['yaw_deg']:.2f}", flush=True)
        icp_results.append(result)
        ply_path.unlink(missing_ok=True)

    # ── 통합 PLY 저장 ────────────────────────────────────────────────────
    if len(np.asarray(combined_pcd.points)) > 0:
        ply_out = result_dir / f"{frame_name}_colored.ply"
        o3d.io.write_point_cloud(str(ply_out), combined_pcd, write_ascii=False)
        log(f"   ✓ 통합 PLY: {ply_out.name}")

    # ── 통합 JSON 저장 ───────────────────────────────────────────────────
    success  = [r for r in icp_results if "error" not in r]
    json_out = result_dir / f"{frame_name}_result.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump({"frame": frame_name,
                   "num_total":   len(icp_results),
                   "num_success": len(success),
                   "instances":   icp_results},
                  f, indent=2, ensure_ascii=False)
    log(f"   ✓ 통합 JSON: {json_out.name}")

    # ── overlay PNG ──────────────────────────────────────────────────────
    overlay_final = draw_picks_on_overlay(bgr_image, picks_2d) if picks_2d \
                    else bgr_image.copy()
    overlay_out = result_dir / f"{frame_name}_overlay.png"
    cv2.imwrite(str(overlay_out), overlay_final)
    log(f"   ✓ overlay PNG: {overlay_out.name}")

    return icp_results, overlay_final

# =============================================================================
# TCP 통신 헬퍼
# =============================================================================
def send_response(conn: socket.socket, payload: dict) -> None:
    conn.sendall((format_response(payload) + "\n").encode("utf-8"))

def format_response(payload: dict) -> str:
    status = payload.get("status")
    if status == "ok":
        picks = payload["picks"]
        parts = ["'ok'", str(len(picks))]
        for pk in picks:
            pp  = pk["position_mm"]
            deg = pk["approach_deg"]
            fit = round(pk["icp_fitness"], 2)
            tup = (round(pp[0], 3), round(pp[1], 3), round(pp[2], 3),
                   round(deg["roll_deg"], 3), round(deg["pitch_deg"], 3),
                   round(deg["yaw_deg"], 3), fit)
            parts.append(str(tup))
        return "{" + ", ".join(parts) + "}"
    elif status in ("no_object", "No"):
        return "{'No'}"
    else:
        msg = payload.get("message", "unknown error")
        return "{" + f"'error', '{msg}'" + "}"

def recv_command(conn: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(1024)
        if not chunk:
            return ""
        buf += chunk
    return buf.decode("utf-8").strip()

# =============================================================================
# 한 프레임 처리 (캡처 → Detection → ICP → payload)
# =============================================================================
def process_one_frame(cam, dirs, frame_idx, cfg_camera,
                      inferencer, cad_pcd, cad_down) -> dict:
    _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n{'─'*70}")
    log(f" [frame_{frame_idx:04d}] 캡처 중... {_now}")

    t0    = time.perf_counter()
    frame = cam.capture()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    meta  = save_capture(frame, dirs, frame_idx, cfg_camera)
    s     = meta["stats"]
    print(f" Captured: {dt_ms:.1f} ms | valid {s['valid_ratio']:.1f}% | "
          f"Z {s['z_min_mm']}~{s['z_max_mm']} mm", flush=True)

    _cap_stat = {"capture_ms": dt_ms, "valid_ratio": s["valid_ratio"],
                 "z_min": s["z_min_mm"] or 0, "z_max": s["z_max_mm"] or 0}

    dt_name    = Path(meta["files"]["intensity"]).stem.replace("intensity_", "")
    frame_name = f"result_{dt_name}"
    gray           = frame.intensity
    pcd_organized  = frame.points_organized.astype(np.float32)
    valid_mask     = frame.valid_mask.astype(bool)

    log(" [Detection]")
    t0 = time.perf_counter()
    summary, inst_plys, bgr_image, bg_pcd = run_detection(
        frame_name, gray, pcd_organized, valid_mask,
        inferencer, dirs["results"])
    det_ms = (time.perf_counter() - t0) * 1000.0
    print(f" 검출: {summary['num_detected']}개 PCD: {summary['num_with_pcd']}개"
          f" ({det_ms:.0f} ms)", flush=True)

    if not inst_plys:
        log(" 브라켓 없음")
        return {"status": "No",
                "_overlay": bgr_image, "_cap_stat": _cap_stat,
                "_info": {"status": "No detection",
                          "det_ms": det_ms, "icp_ms": 0, "picks": []}}

    log(" [ICP]")
    t0 = time.perf_counter()
    icp_results, final_overlay = run_icp_for_frame(
        inst_plys, cad_pcd, cad_down, dirs["results"],
        frame_name, bgr_image, bg_pcd=bg_pcd)
    icp_ms  = (time.perf_counter() - t0) * 1000.0
    success = [r for r in icp_results if "error" not in r]
    n_fail  = len(icp_results) - len(success)
    log(f" ICP: 성공 {len(success)}개 실패 {n_fail}개 ({icp_ms:.0f} ms)")

    if not success:
        return {"status": "No",
                "_overlay": final_overlay, "_cap_stat": _cap_stat,
                "_info": {"status": "No detection",
                          "det_ms": det_ms, "icp_ms": icp_ms, "picks": []}}

    picks = [{"position_mm":  r["pick_point"]["position_mm"],
               "approach_deg": r["pick_point"]["approach_deg"],
               "icp_fitness":  r["icp_fitness"]}
             for r in success]

    for i, pk in enumerate(picks):
        pp  = pk["position_mm"]
        deg = pk["approach_deg"]
        fit = pk["icp_fitness"]
        log(f" #{i} 위치: ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm fit={fit:.2f}"
            f" roll={deg['roll_deg']:.2f} pitch={deg['pitch_deg']:.2f}"
            f" yaw={deg['yaw_deg']:.2f}")

    return {"status": "ok", "picks": picks,
            "_overlay": final_overlay, "_cap_stat": _cap_stat,
            "_info": {"status": "Done",
                      "det_ms": det_ms, "icp_ms": icp_ms, "picks": picks}}

# =============================================================================
# argparse
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="빈피킹 TCP 서버")
    p.add_argument("--config",  type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--out",     type=Path, default=ROOT / "data" / "captures" / "live")
    p.add_argument("--warmup",  type=int,  default=3)
    p.add_argument("--host",    type=str,  default="0.0.0.0",
                   help="TCP 바인드 주소 (기본: 0.0.0.0 = 모든 인터페이스)")
    p.add_argument("--port",    type=int,  default=29999,
                   help="TCP 포트 (기본: 29999)")
    return p.parse_args()

# =============================================================================
# main
# =============================================================================
def main():
    global _logger
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_camera = cfg.get("camera", {})

    dirs    = setup_dirs(args.out)
    _logger = setup_file_logger(args.out)

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    log(f"모델 로드: {CHECKPOINT_PATH.name}")
    inferencer = RTMDetInferencer(
        config=str(CONFIG_PATH),
        checkpoint=str(CHECKPOINT_PATH),
        score_threshold=SCORE_THRESHOLD,
    )

    # ── CAD 로드 ───────────────────────────────────────────────────────────
    log(f"CAD 로드: {CAD_PATH.name}")
    cad_pcd  = load_cad_as_pcd(CAD_PATH)
    cad_down = cad_pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    log(f"CAD 포인트 수: {len(np.asarray(cad_pcd.points))} "
        f"(다운샘플: {len(np.asarray(cad_down.points))})")

    # ── 카메라 초기화 ──────────────────────────────────────────────────────
    log("카메라 초기화 중...")
    cam = create_camera(cfg_camera)
    log(f"카메라 IP: {getattr(cam, 'ip', 'N/A')}")

    log(f"워밍업 {args.warmup}프레임...")
    for _ in range(args.warmup):
        cam.capture()

    # 모니터 창 초기화
    dummy = cam.capture()
    init_monitor(dummy.height, dummy.width)

    # ── TCP 서버 ───────────────────────────────────────────────────────────
    frame_idx = 0
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    log(f"\nTCP 서버 대기 중: {args.host}:{args.port}")
    log("회전 구속: "
        f"roll [{ICP_ROLL_RANGE[0]}, {ICP_ROLL_RANGE[1]}]°  "
        f"pitch [{ICP_PITCH_RANGE[0]}, {ICP_PITCH_RANGE[1]}]°  "
        f"yaw [{ICP_YAW_RANGE[0]}, {ICP_YAW_RANGE[1]}]°")

    try:
        while True:
            conn, addr = srv.accept()
            log(f"연결: {addr}")
            try:
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:           # ESC
                        cv2.destroyAllWindows()

                    cmd = recv_command(conn)
                    if not cmd:
                        log("연결 종료 (클라이언트)")
                        break
                    log(f"수신: {cmd!r}")

                    if cmd.upper() == "QUIT":
                        log("QUIT 수신 → 서버 종료")
                        send_response(conn, {"status": "ok", "picks": []})
                        conn.close()
                        srv.close()
                        cv2.destroyAllWindows()
                        return

                    elif cmd.upper() in ("CAPTURE", "C"):
                        frame_idx += 1
                        _ts = datetime.now().strftime("%H:%M:%S")

                        # 모니터: Processing 표시
                        show_monitor(
                            np.zeros((dummy.height, dummy.width, 3), dtype=np.uint8),
                            {"status": "Processing", "frame_idx": frame_idx,
                             "timestamp": _ts, "picks": []})

                        try:
                            payload = process_one_frame(
                                cam, dirs, frame_idx, cfg_camera,
                                inferencer, cad_pcd, cad_down)
                            send_response(conn, payload)

                            overlay = payload.get("_overlay")
                            cs      = payload.get("_cap_stat", {})
                            info    = payload.get("_info", {})
                            info.update({"frame_idx": frame_idx,
                                         "timestamp": _ts,
                                         **cs})
                            if overlay is not None:
                                show_monitor(overlay, info)

                        except Exception as e:
                            log(f"ERROR: {e}")
                            send_response(conn,
                                          {"status": "error", "message": str(e)})
                    else:
                        log(f"알 수 없는 명령: {cmd!r}")
                        send_response(conn,
                                      {"status": "error",
                                       "message": f"unknown command: {cmd}"})
            finally:
                conn.close()
    except KeyboardInterrupt:
        log("\nKeyboardInterrupt → 종료")
    finally:
        srv.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
