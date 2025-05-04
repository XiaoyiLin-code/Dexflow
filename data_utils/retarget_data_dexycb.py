import sys
import os
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
from pathlib import Path  # 确保导入 Path
import numpy as np
from collections import defaultdict
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.hand_model import create_hand_model
from transforms3d.euler import euler2mat
import plotly.graph_objects as go
import torch
from pytransform3d import rotations
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien
from pytransform3d import transformations as pt

from config.path_config import Path_config

def compute_rotation_matrix1(from_vec1, from_vec2, from_vec3, to_vec1, to_vec2, to_vec3):
    # 单位化所有向量
    from_vec1 = from_vec1 / np.linalg.norm(from_vec1)
    from_vec2 = from_vec2 / np.linalg.norm(from_vec2)
    from_vec3 = from_vec3 / np.linalg.norm(from_vec3)
    to_vec1 = to_vec1 / np.linalg.norm(to_vec1)
    to_vec2 = to_vec2 / np.linalg.norm(to_vec2)
    to_vec3 = to_vec3 / np.linalg.norm(to_vec3)

    # 构造初始坐标系：x = from_vec1, y = from_vec2, z = from_vec3
    from_basis = np.stack((from_vec1, from_vec2, from_vec3), axis=0)

    # 构造目标坐标系：x = to_vec1, y = to_vec2, z = to_vec3
    to_basis = np.stack((to_vec1, to_vec2, to_vec3), axis=0)

    # 计算右乘旋转矩阵 R_{AB}，使得：from = R_{AB} @ to
    R_matrix = from_basis.T @ to_basis  # 注意这里是 from_basis.T @ to_basis
    print(to_basis, from_basis@R_matrix)

    # 将旋转矩阵转换为 Rotation 对象并返回
    rotation = R.from_matrix(R_matrix)
    r= rotation.as_euler('xyz', degrees=True)[0]
    p= rotation.as_euler('xyz', degrees=True)[1]
    y= rotation.as_euler('xyz', degrees=True)[2]
    Rx=np.array([[1, 0, 0],
                   [0, np.cos(np.radians(r)), -np.sin(np.radians(r))],
                   [0, np.sin(np.radians(r)), np.cos(np.radians(r))]])
    Ry=np.array([[np.cos(np.radians(p)), 0, np.sin(np.radians(p))],
                   [0, 1, 0],
                   [-np.sin(np.radians(p)), 0, np.cos(np.radians(p))]])
    Rz=np.array([[np.cos(np.radians(y)), -np.sin(np.radians(y)), 0],
                     [np.sin(np.radians(y)), np.cos(np.radians(y)), 0],
                     [0, 0, 1]])
    print(from_basis@Rz@Ry@Rx)
    print(from_vec1@Rz@Ry@Rx)
    print(from_vec2@Rz@Ry@Rx)
    print(from_vec3@Rz@Ry@Rx)
    return rotation
def compute_rotation_matrix(from_vec1, from_vec2, to_vec1, to_vec2):
    # 单位化所有向量
    from_vec1 = from_vec1 / np.linalg.norm(from_vec1)
    from_vec2 = from_vec2 / np.linalg.norm(from_vec2)
    to_vec1 = to_vec1 / np.linalg.norm(to_vec1)
    to_vec2 = to_vec2 / np.linalg.norm(to_vec2)

    # 构造初始坐标系：z = from_vec1，y = from_vec2，x = y × z
    from_x = np.cross(from_vec2, from_vec1)
    from_basis = np.stack((from_x, from_vec2, from_vec1), axis=1)  # 列向量为基向量

    # 构造目标坐标系：z = to_vec1，y = to_vec2，x = y × z
    to_x = np.cross(to_vec2, to_vec1)
    to_basis = np.stack((to_x, to_vec2, to_vec1), axis=1)

    # 旋转矩阵 R，使得：to = R @ from
    R_matrix = to_basis @ from_basis.T
    rotation = R.from_matrix(R_matrix)
    return rotation.as_matrix(),rotation

def validate_rotation(R, from_vec1, from_vec2, to_vec1, to_vec2):
    # 验证旋转矩阵是否正确
    result_vec1 = R @ from_vec1
    result_vec2 = R @ from_vec2
    
    print(f"验证结果 (R @ from_vec1): {result_vec1}")
    print(f"目标 to_vec1: {to_vec1}")
    print(f"验证结果 (R @ from_vec2): {result_vec2}")
    print(f"目标 to_vec2: {to_vec2}")

    # 计算向量间的余弦相似度，检查是否接近 1（即向量是否对齐）
    dot_product_vec1 = np.dot(result_vec1, to_vec1)
    dot_product_vec2 = np.dot(result_vec2, to_vec2)

    if np.isclose(dot_product_vec1, 1.0) and np.isclose(dot_product_vec2, 1.0):
        print("验证成功：旋转矩阵使得向量正确对齐。")
    else:
        print("验证失败：旋转矩阵未能使向量正确对齐。")




def rotation_matrix1(v1, v2):
    # 归一化向量
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 使用 scipy 计算旋转矩阵
    rotation = R.align_vectors([v2], [v1])[0]  # 返回一个 Rotation 对象

    # 获取旋转矩阵
    return rotation.as_matrix(), rotation
def rpy_to_normal(rpy):
    """
    将 Roll-Pitch-Yaw 欧拉角（Z-Y-X 顺序）转换为法向量
    :param rpy: roll, pitch, yaw（弧度）
    :return: 单位法向量 (shape: (3,))
    """
    # 提取欧拉角
    roll, pitch, yaw = rpy

    # 计算旋转矩阵
    R = euler2mat(roll, pitch, yaw)

    # 提取法向量
    normal = R[:, 2]  # Z 轴方向

    # 单位化法向量
    normal /= np.linalg.norm(normal)

    return normal

def normal_to_rpy(normal):
    """
    将法向量转换为 Roll-Pitch-Yaw 欧拉角（Z-Y-X 顺序）
    :param normal: 单位法向量 (shape: (3,))
    :return: roll, pitch, yaw（弧度）
    """
    # 确保输入是单位向量

    
    # 定义世界坐标系参考轴（假设 Y 轴为 "up" 方向）
    WORLD_UP = np.array([0, 0.0, 1.0])
    
    # 避免法向量与世界坐标系 Y 轴重合时的奇点
    if np.abs(np.dot(normal, WORLD_UP)) > 0.999:
        # 如果法向量接近世界 Y 轴，改用 X 轴作为参考
        right = np.array([1.0, 0, 0])
    else:
        # 计算右方向（X 轴）：WORLD_UP × normal
        right = np.cross(WORLD_UP, normal)
        right /= np.linalg.norm(right)
    
    # 计算前方向（Y 轴）：normal × right
    forward = np.cross(normal, right)
    forward /= np.linalg.norm(forward)
    
    # 构造旋转矩阵（从世界坐标系到物体坐标系）
    # 列向量为 [right, forward, normal]
    R = np.column_stack((right, forward, normal))
    
    # 提取欧拉角（Z-Y-X 顺序）
    yaw = np.arctan2(R[1, 0], R[0, 0])         # ψ (绕 Z 轴)
    pitch = np.arcsin(-R[2, 0])               # θ (绕 Y 轴)
    roll = np.arctan2(R[2, 1], R[2, 2])       # φ (绕 X 轴)
    
    return [roll, pitch, yaw]



def pose_to_matrix(pose):
    quat = pose[:4]  
    trans = pose[4:]
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    matrix[:3, 3] = trans
    return matrix

def matrix_to_pose(matrix):
    quat = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([[quat[3]], quat[:3], matrix[:3, 3]])

def apply_extrinsic_to_object(object_pose, T_world_cam):
    T_obj_world = pose_to_matrix(object_pose)
    T_obj_cam = T_world_cam @ T_obj_world
    return matrix_to_pose(T_obj_cam)  
def axis_angle_to_matrix(axis_angle):
    device = axis_angle.device if isinstance(axis_angle, torch.Tensor) else 'cpu'
    
    if isinstance(axis_angle, torch.Tensor):
        theta = torch.norm(axis_angle)
        if theta < 1e-30:
            return torch.eye(3, device=device) 
        axis = axis_angle / theta
    else:
        theta = np.linalg.norm(axis_angle)
        if theta < 1e-30:
            return torch.eye(3, device=device)
        axis = torch.tensor(axis_angle / theta, device=device)
    K = torch.tensor([
        [0, -axis[2].item(), axis[1].item()],
        [axis[2].item(), 0, -axis[0].item()],
        [-axis[1].item(), axis[0].item(), 0]
    ], device=device)
    eye = torch.eye(3, device=device)
    R_mat = eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
    return R_mat

def matrix_to_axis_angle(rotation_matrix):
    device = rotation_matrix.device
    trace = torch.trace(rotation_matrix)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0)) 
    
    if theta < 1e-30:
        return torch.zeros(3, device=device)
    
    axis = torch.stack([
        rotation_matrix[2,1] - rotation_matrix[1,2],
        rotation_matrix[0,2] - rotation_matrix[2,0],
        rotation_matrix[1,0] - rotation_matrix[0,1]
    ]).to(device) / (2 * torch.sin(theta))
    
    return theta * axis

def apply_extrinsic_to_mano(hand_pose, T_world_cam):
    device = hand_pose.device
    T_world_cam = torch.tensor(T_world_cam, 
                             dtype=hand_pose.dtype, 
                             device=device) 
    rot_aa = hand_pose[:3]
    trans = hand_pose[-3:]
    T_hand_world = torch.eye(4, device=device)
    T_hand_world[:3, :3] = axis_angle_to_matrix(rot_aa)
    T_hand_world[:3, 3] = trans
    T_hand_cam = T_world_cam @ T_hand_world
    new_rot_aa = matrix_to_axis_angle(T_hand_cam[:3, :3])
    new_trans = T_hand_cam[:3, 3]
    
    return torch.cat([new_rot_aa, hand_pose[3:-3], new_trans])



def apply_object_pose_transformation(object_pc, object_mesh, object_pose):
    # 解析 object_pose（四元数顺序为 wxyz）
    translation = object_pose.p  # 平移向量 [x, y, z]
    quaternion_wxyz = object_pose.q  # 四元数 [w, x, y, z]
    # translation = np.array(object_pose[-3:])  # 平移向量 [x, y, z]
    # quaternion = np.array(object_pose[:-3])  # 四元数 [qx, qy, qz, qw] 
    # 将四元数从 wxyz 转换为 SciPy 需要的 xyzw 格式
    quaternion_xyzw = np.roll(quaternion_wxyz, shift=-1)  # 从 [w, x, y, z] 转为 [x, y, z, w]
    
    # 创建旋转矩阵
    rotation = R.from_quat(quaternion_xyzw).as_matrix()

    # 确保输入点云是 NumPy 数组
    if isinstance(object_pc, torch.Tensor):
        object_pc = object_pc.cpu().numpy()

    # 应用旋转和平移
    transformed_pc = (rotation @ object_pc.T).T + translation  # 点云变换
    transformed_vertices = (rotation @ object_mesh.vertices.T).T + translation  # 网格顶点变换

    # 更新网格
    transformed_mesh = object_mesh.copy()
    transformed_mesh.vertices = transformed_vertices

    # 将点云转回 PyTorch Tensor
    return torch.tensor(transformed_pc, dtype=torch.float32), transformed_mesh


def visualize_point_clouds(mano_mesh, object_pc):
    mano_vertices = np.array(mano_mesh.vertices)
    if isinstance(object_pc, torch.Tensor):
        object_pc = object_pc.cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mano_vertices[:, 0], mano_vertices[:, 1], mano_vertices[:, 2], c='blue', s=1, label='Mano Mesh')
    ax.scatter(object_pc[:, 0], object_pc[:, 1], object_pc[:, 2], c='red', s=1, label='Object Point Cloud')
    ax.set_title("Mano Mesh and Object Point Cloud", fontsize=15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}



import yaml 


class DexYCB_Dataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        select_idx: np.ndarray = None,
    ):
        self.batch_size = batch_size
        self.select_idx = select_idx
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.object_meshes = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hand_name = "mano"
        self.hand_type = ["right"]
        self.hand = create_hand_model(self.hand_name, self.device)

        self.object_names = []

    
        self._mano_model_dir = Path(Path_config.get_third_party_path()) / "manopth"
        self._calib_dir = self._mano_model_dir / "calibration"
        self._model_dir = self._mano_model_dir / "models"
        self._graspdata_dir = "/home/lightcone/workspace/DRO-retarget/Noise-learn/data/dex_ycb"
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    

        self._prepare_data()
        self._load_camera_parameters()

    def _prepare_data(self):
        subject_dirs = sorted(
            Path(self._graspdata_dir).glob("*/"),
            key=lambda p: int(p.stem.split('-')[-1])
        )
        self._subject_dirs = []
        self._session_dirs = []
        data_idx = 0
        for subject_dir in subject_dirs:
            # session_dirs = sorted(
            #     subject_dir.glob("*/"),
            #     key=lambda x: int(x.stem.split('-')[-1])  # 假设目录名格式如"20200709-subject-01"
            # )

            session_dirs=subject_dir.glob("*/")
            self._subject_dirs.append(subject_dir.name)
            for session_dir in session_dirs:
                if data_idx not in self.select_idx:
                    data_idx += 1
                    continue
                meta_file = session_dir / "meta.yml"
                if not meta_file.exists():               
                    print(f"No meta.yml file found in {session_dir}. Skipping...")
                    continue
                with open(meta_file, 'r') as f:
                    meta_data = yaml.safe_load(f)
                    if meta_data["mano_sides"] != self.hand_type:
                        continue
                    self.metadata[subject_dir.name][session_dir.name] = meta_data
                self._session_dirs.append(session_dir.name)
                mano_pose_file = session_dir / "pose.npz"
                if not mano_pose_file.exists():
                    print(f"No pose.npz file found in {session_dir}. Skipping...")
                    continue
                mano_pose = np.load(mano_pose_file)

                pose_m = mano_pose["pose_m"]
                pose_y = mano_pose["pose_y"]

                pose_m_flat = pose_m[:, 0, :] 
                mask = ~(pose_m_flat == 0).all(axis=1)  
                filtered_pose_m = pose_m[mask, :, :]  
                filtered_pose_y = pose_y[mask, :, :] 
                moving_objects_mask = (filtered_pose_y.max(axis=0) != filtered_pose_y.min(axis=0)).any(axis=1)
                filtered_pose_y = filtered_pose_y[:, moving_objects_mask, :] 

                self.metadata[subject_dir.name][session_dir.name]["mano_pose"] = filtered_pose_m
                self.metadata[subject_dir.name][session_dir.name]["obj_pose"] = filtered_pose_y


                moving_ycb_ids = [
                    self.metadata[subject_dir.name][session_dir.name]["ycb_ids"][i]
                    for i, is_moving in enumerate(moving_objects_mask) if is_moving
                ]

                for obj_id in moving_ycb_ids:
                    self.object_names.append(YCB_CLASSES[obj_id])
                    obj_path = self._mano_model_dir / "models" / YCB_CLASSES[obj_id] / "textured_simple.obj"
                    self.metadata[subject_dir.name][session_dir.name][f"{YCB_CLASSES[obj_id]}_file"] = obj_path
                    if obj_path.exists():
                        try:
                            mesh = trimesh.load_mesh(obj_path, file_type='obj')
                            self.metadata[subject_dir.name][session_dir.name][f"{YCB_CLASSES[obj_id]}_mesh"] = mesh
                            self.object_meshes[f"{YCB_CLASSES[obj_id]}"] = mesh
                        except Exception as e:
                            print(f"Failed to load {obj_id}: {obj_path}. Error: {e}")
                    else:
                        print(f"File not found: {obj_path}")
                data_idx += 1
            break
    def _load_camera_parameters(self):
        extrinsics = {}
        intrinsics = {}
        cali_dirs = sorted(
            [d for d in self._calib_dir.iterdir() if d.stem.startswith("extrinsics")],
            key=lambda x: x.stem  
        )
        for cali_dir in cali_dirs:
            if not cali_dir.stem.startswith("extrinsics"):
                continue
            extrinsic_file = cali_dir / "extrinsics.yml"
            name = cali_dir.stem[len("extrinsics_") :]
            with extrinsic_file.open(mode="r") as f:
                extrinsic = yaml.load(f, Loader=yaml.FullLoader)
            extrinsics[name] = extrinsic

        intrinsic_dir = self._calib_dir / "intrinsics"
        for intrinsic_file in intrinsic_dir.iterdir():
            with intrinsic_file.open(mode="r") as f:
                intrinsic = yaml.load(f, Loader=yaml.FullLoader)
            name = intrinsic_file.stem.split("_")[0]
            x = intrinsic["color"]
            camera_mat = np.array(
                [[x["fx"], 0.0, x["ppx"]], [0.0, x["fy"], x["ppy"]], [0.0, 0.0, 1.0]]
            )
            intrinsics[name] = camera_mat

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics


    def __getitem__(self, index):
            hand_name_batch = []
            object_name_batch = []
            object_pc_batch = []
            target_q_batch = []
            mano_mesh_batch = []
            mano_joint_batch = []
            wrist_quat_batch = []
            object_mesh_batch = []
            object_pose_batch = []
            dir_index_batch = []
            normal_batch = []
            normal_batch2 = []

            for idx in range(self.batch_size):
                hand_name_batch.append(self.hand_name)
                object_index = (index * self.batch_size + idx)
                object_name = self.object_names[object_index]
                object_name_batch.append(object_name)
                cumulative_lengths = np.cumsum([len(self.metadata[dir]) for dir in self._subject_dirs])
                global_index = index * self.batch_size + idx
                subject_dir_index = np.searchsorted(cumulative_lengths, global_index, side='right')
                session_dir_index = global_index
                data=self.metadata[self._subject_dirs[subject_dir_index]][self._session_dirs[session_dir_index]]
                hand_poses = data['mano_pose']
                dir_index_batch.append((subject_dir_index, session_dir_index))
                
                
                extrinsic_key = data['extrinsics']
                extrinsic_mat = np.array(self.extrinsics[extrinsic_key]["extrinsics"]["apriltag"]).reshape([3, 4])
                extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)
                pose_vec = pt.pq_from_transform(extrinsic_mat)
                self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()


                start_frame = 40
                end_frame = min(hand_poses.shape[0], start_frame + 5)
                for i in range(start_frame,end_frame):
                    hand_pose = torch.tensor(hand_poses[i].squeeze(0)).to(self.device)

                    object_pose = data['obj_pose'][i].squeeze(0) 
                    object_pose = self.camera_pose * sapien.Pose(object_pose[4:], np.concatenate([object_pose[3:4], object_pose[:3]])) 
                    target_q_batch.append(hand_pose)
                    self.hand.set_parameters(hand_pose.unsqueeze(0),data_origin="ycb")
                    self.hand.camera_to_world(self.camera_pose)
                    object_mesh = data[f"{object_name}_mesh"]
                    object_pc = object_mesh.sample(5000)  
                    object_pc = torch.tensor(object_pc, dtype=torch.float)
                    


                    transformed_pc, transformed_mesh = apply_object_pose_transformation(object_pc, object_mesh, object_pose)
                    object_pc_batch.append(transformed_pc)
                    object_mesh_batch.append(transformed_mesh)

                    mano_mesh, mano_joint = self.hand.get_trans_trimesh_data(i=0, pose=None)
                    mano_mesh_batch.append(mano_mesh)
                    mano_joint=torch.tensor(mano_joint, dtype=torch.float32)
                    mano_joint_batch.append(mano_joint)
                    wrist_quat = hand_pose[:3]  
                    rot_x_90= torch.tensor(euler2mat(0, 0, np.pi /2), dtype=torch.float32)
                    rot_y_90= torch.tensor(euler2mat(0, -np.pi/2, 0), dtype=torch.float32)
                    rot_z_90= torch.tensor(euler2mat(np.pi / 2, 0, 0), dtype=torch.float32)
                    wrist_quat = torch.tensor(axis_angle_to_matrix(wrist_quat).cpu(), dtype=torch.float32)
                    wrist_quat = matrix_to_axis_angle(wrist_quat)
                    # wrist_quat_batch.append(wrist_quat)

                    translation = object_pose.p
                    quaternion = object_pose.q
                    whole_pose = np.concatenate([quaternion, translation])
                    whole_pose = torch.tensor(whole_pose, dtype=torch.float32)
                    object_pose_batch.append(whole_pose)
                    face_indices = [38, 122, 118, 117, 119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234, 92]
                    face_vertices = mano_mesh.vertices[face_indices]  # 形状为 (16, 3)
                    centroid = np.mean(face_vertices, axis=0)
                    points_centered = face_vertices - centroid
                    U, S, Vt = np.linalg.svd(points_centered)
                    normal = Vt[2] 
                    normal = normal / np.linalg.norm(normal)
                    from_vec1 = np.array([0, 0, 1])  # 初始向量1 (X轴)
                    from_vec2 = np.array([1, 0, 0])  # 初始向量2 (Y轴)
                    from_vec3 = np.array([0, 1, 0])  # 初始向量3 (Z轴)

                    to_vec1 = normal
                    to_vec1=to_vec1/np.linalg.norm(to_vec1)

                    to_vec2 = mano_mesh.vertices[92]-mano_mesh.vertices[79]
                    to_vec2 = to_vec2 - np.dot(to_vec2, to_vec1) * to_vec1
                    to_vec2 = to_vec2 / np.linalg.norm(to_vec2)
                    to_vec3 = np.cross(to_vec1, to_vec2)  # 目标向量3（从目标向量1和目标向量2的叉积得到）

                    rotation = compute_rotation_matrix1(from_vec1, from_vec2, from_vec3, to_vec1, to_vec2, to_vec3)
                    
                    wrist_quat_batch.append(rotation)
                    normal_batch.append(to_vec1)
                    normal_batch2.append(to_vec2)


            target_q_batch = torch.stack(target_q_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            mano_joint_batch = torch.stack(mano_joint_batch) 

            import gc
            unreachable_objects = gc.collect()
            return {
                "hand_pose": target_q_batch,
                "object_pcs": object_pc_batch,
                "mano_mesh": mano_mesh_batch,
                "mano_joint": mano_joint_batch,
                "wrist_quat": wrist_quat_batch,
                "object_meshes": object_mesh_batch,
                "object_pose": object_pose_batch,
                "object_name": object_name_batch,
                "dir_index": dir_index_batch,
                "normal": normal_batch,
                "normal2": normal_batch2,
            }
    


