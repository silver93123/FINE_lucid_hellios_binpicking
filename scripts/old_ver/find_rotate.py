import open3d as o3d
import numpy as np
import copy

ROOT = "/home/silver/binpicking_vision/RTM_test"

mesh = o3d.io.read_triangle_mesh(f"{ROOT}/data/cad/bracket_v2.stl")
mesh.scale(1/1000, center=mesh.get_center())
mesh.compute_vertex_normals()
cad_pcd = mesh.sample_points_poisson_disk(30000)

scene_pcd = o3d.io.read_point_cloud(
    f"{ROOT}/data/inference_results/20260519_0000/frame_0001_obj0.ply"
)

def test_rotation(rx_deg, ry_deg, rz_deg):
    """CAD를 자체 중심 기준 회전 → scene 중심으로 이동 후 시각화."""
    cad = copy.deepcopy(cad_pcd)
    center = np.asarray(cad.get_center())

    # 각 축 회전행렬
    def Rx(d):
        a = np.radians(d)
        return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    def Ry(d):
        a = np.radians(d)
        return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    def Rz(d):
        a = np.radians(d)
        return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

    R = Rz(rz_deg) @ Ry(ry_deg) @ Rx(rx_deg)

    # 1. CAD 자체 중심 기준으로 회전
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    T_rot[:3, 3] = center - R @ center   # 중심 고정 회전

    # 2. scene 중심으로 평행이동
    cad.transform(T_rot)
    tgt_c = np.asarray(scene_pcd.get_center())
    src_c = np.asarray(cad.get_center())
    cad.translate(tgt_c - src_c)

    cad.paint_uniform_color([0.1, 0.9, 0.3])
    s = copy.deepcopy(scene_pcd)
    s.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([cad, s],
        window_name=f"Rx={rx_deg} Ry={ry_deg} Rz={rz_deg}")

# 일단 X축 -90도 시도
test_rotation(rx_deg=-90, ry_deg=90, rz_deg=100)