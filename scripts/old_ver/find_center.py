import open3d as o3d
import numpy as np

ROOT = "/home/silver/binpicking_vision/RTM_test"

mesh = o3d.io.read_triangle_mesh(f"{ROOT}/data/cad/bracket_v2.stl")
mesh.scale(1/1000, center=np.zeros(3))

def Rx(d): a=np.radians(d); return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
def Ry(d): a=np.radians(d); return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def Rz(d): a=np.radians(d); return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

R = Rz(90) @ Ry(90) @ Rx(-90)
center = np.asarray(mesh.get_center())
T = np.eye(4); T[:3,:3] = R; T[:3,3] = center - R @ center
mesh.transform(T)
mesh.compute_triangle_normals()

triangles  = np.asarray(mesh.triangles)
vertices   = np.asarray(mesh.vertices)
tri_normals = np.asarray(mesh.triangle_normals)

# 노말의 Z 성분이 강한 면 = 수평 면
# |nz| > 0.9 이면 거의 수평
horizontal_mask = np.abs(tri_normals[:, 2]) > 0.9
print(f"수평 삼각형 수: {horizontal_mask.sum()} / {len(triangles)}")

# 수평 면의 중심점 계산
horiz_tris = triangles[horizontal_mask]
horiz_centers = vertices[horiz_tris].mean(axis=1)  # 각 삼각형 중심
horiz_normals = tri_normals[horizontal_mask]

# nz > 0 (위를 향하는 면) vs nz < 0 (아래를 향하는 면) 분리
up_mask   = horiz_normals[:, 2] > 0
down_mask = horiz_normals[:, 2] < 0

print(f"\n위 향하는 수평면 (nz>0): {up_mask.sum()}개")
if up_mask.sum() > 0:
    up_pts = horiz_centers[up_mask]
    print(f"  Z 평균: {up_pts[:,2].mean():.4f} m")
    print(f"  중심:   X={up_pts[:,0].mean():.4f}  Y={up_pts[:,1].mean():.4f}  Z={up_pts[:,2].mean():.4f}")

print(f"\n아래 향하는 수평면 (nz<0): {down_mask.sum()}개")
if down_mask.sum() > 0:
    dn_pts = horiz_centers[down_mask]
    print(f"  Z 평균: {dn_pts[:,2].mean():.4f} m")
    print(f"  중심:   X={dn_pts[:,0].mean():.4f}  Y={dn_pts[:,1].mean():.4f}  Z={dn_pts[:,2].mean():.4f}")

# 시각화 — 수평면 빨강, 나머지 초록
pcd = mesh.sample_points_poisson_disk(30000)
pts = np.asarray(pcd.points)

# 각 포인트가 어느 삼각형에 속하는지 근사 — 수평면 Z 높이로 판별
if up_mask.sum() > 0:
    target_z = horiz_centers[up_mask][:,2].mean()
else:
    target_z = horiz_centers[down_mask][:,2].mean()

colors = np.tile([0.1, 0.9, 0.3], (len(pts), 1))
near_z = np.abs(pts[:,2] - target_z) < 0.002
colors[near_z] = [1.0, 0.1, 0.1]
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="수평면 확인")