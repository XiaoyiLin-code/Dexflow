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

def apply_extrinsic_to_object(object_pose, extrinsic_mat):
    """
    将相机外参应用到物体的旋转（四元数）和平移。

    参数:
        object_pose (np.ndarray): 物体的全局姿态，前 4 位是四元数 [w, x, y, z]，后 3 位是平移向量。
        extrinsic_mat (np.ndarray): 相机外参矩阵，形状为 (4, 4)。

    返回:
        np.ndarray: 更新后的物体姿态，格式为 [w, x, y, z, tx, ty, tz]。
    """
    # 提取四元数和平移
    obj_quat = object_pose[:4]  # 四元数 (w, x, y, z)
    obj_trans = object_pose[4:]  # 平移向量

    # 将四元数转换为旋转矩阵
    rotation_matrix = R.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()

    # 构造物体的齐次变换矩阵
    T_object = np.eye(4)
    T_object[:3, :3] = rotation_matrix
    T_object[:3, 3] = obj_trans

    # 应用外参矩阵
    T_camera_object = extrinsic_mat @ T_object

    # 提取新的旋转和平移
    new_rotation_matrix = T_camera_object[:3, :3]
    new_translation_vector = T_camera_object[:3, 3]

    # 将旋转矩阵转换为四元数
    new_quaternion = R.from_matrix(new_rotation_matrix).as_quat()  # [x, y, z, w]

    # 调整四元数格式为 [w, x, y, z]
    new_quaternion = [new_quaternion[3], new_quaternion[0], new_quaternion[1], new_quaternion[2]]

    # 组合新的姿态
    new_object_pose = np.concatenate([new_quaternion, new_translation_vector])

    return new_object_pose

def apply_extrinsic_to_mano(hand_pose, extrinsic_mat):
    """
    将相机外参应用到 MANO 数据的旋转和平移部分。

    参数:
        hand_pose (torch.Tensor): MANO 的 hand_pose，长度为 51（仅用到前 3 位和最后 3 位）
        extrinsic_mat (np.ndarray): 相机外参矩阵，形状为 (4, 4)

    返回:
        torch.Tensor: 更新后的 hand_pose。
    """
    # 提取 MANO 的全局旋转和全局平移
    mano_rot = hand_pose[:3].numpy()  # 前三位（全局旋转，axis-angle）
    mano_trans = hand_pose[-3:].numpy()  # 最后三位（全局平移）

    # 构造 MANO 的齐次变换矩阵
    rotation_matrix = R.from_rotvec(mano_rot).as_matrix()  # 旋转矩阵
    T_mano = np.eye(4)
    T_mano[:3, :3] = rotation_matrix
    T_mano[:3, 3] = mano_trans

    # 应用外参矩阵
    T_camera = extrinsic_mat @ T_mano

    # 提取新的旋转和平移
    new_rotation_matrix = T_camera[:3, :3]
    new_translation_vector = T_camera[:3, 3]

    # 将旋转矩阵转换回 axis-angle 格式
    new_rotation_vector = R.from_matrix(new_rotation_matrix).as_rotvec()

    # 更新 hand_pose 的旋转和平移部分
    hand_pose[:3] = torch.tensor(new_rotation_vector, dtype=torch.float32)
    hand_pose[-3:] = torch.tensor(new_translation_vector, dtype=torch.float32)

    return hand_pose


def apply_object_pose_transformation(object_pc, object_mesh, object_pose):
    # 解析 object_pose
    translation = np.array(object_pose[-3:])  # 平移向量 [x, y, z]
    quaternion = np.array(object_pose[:-3])  # 四元数 [qx, qy, qz, qw]

    # 构建旋转矩阵
    rotation = R.from_quat(quaternion).as_matrix()  # 3x3 旋转矩阵

    # 对点云进行旋转和平移
    if isinstance(object_pc, torch.Tensor):
        object_pc = object_pc.cpu().numpy()  # 转为 numpy
    transformed_pc = (rotation @ object_pc.T).T + translation  # 点云的刚体变换

    # 对网格的顶点进行旋转和平移
    transformed_vertices = (rotation @ object_mesh.vertices.T).T + translation

    # 更新网格
    transformed_mesh = object_mesh.copy()
    transformed_mesh.vertices = transformed_vertices

    # 返回变换后的点云和网格
    return torch.tensor(transformed_pc, dtype=torch.float32), transformed_mesh


def visualize_point_clouds(mano_mesh, object_pc):
    """
    可视化点云数据（mano_mesh 和 object_pc）
    :param mano_mesh: 手部网格 (Trimesh 点云数据)
    :param object_pc: 对象点云 (torch.Tensor 或 numpy.ndarray)
    """
    # 获取 mano_mesh 的顶点坐标
    mano_vertices = np.array(mano_mesh.vertices)
    
    # 如果 object_pc 是 torch.Tensor，转换为 numpy
    if isinstance(object_pc, torch.Tensor):
        object_pc = object_pc.cpu().numpy()

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制手部点云
    ax.scatter(mano_vertices[:, 0], mano_vertices[:, 1], mano_vertices[:, 2], c='blue', s=1, label='Mano Mesh')

    # 绘制对象点云
    ax.scatter(object_pc[:, 0], object_pc[:, 1], object_pc[:, 2], c='red', s=1, label='Object Point Cloud')

    # 添加标签与图例
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



import yaml  # 用于读取 YAML 文件


class DexYCB_Dataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        hand_name: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 5000,
        object_pc_type: str = 'random',
        noise_level: str = 'pointcloud',
        mano_root: str = None
    ):
        self.noise_level = noise_level
        self.batch_size = batch_size
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.object_meshes = {}
        device = 'cpu'

        self.hand_name = "mano"
        self.hand_type = ["right"]
        self.hand = create_hand_model(self.hand_name, torch.device('cpu'))

        self.object_names = []

        self._data_dir = Path("/home/lightcone/workspace/DRO-retarget/Noise-learn/thirdparty/manopth")
        self._calib_dir = self._data_dir / "calibration"
        self._model_dir = self._data_dir / "models"
        self._graspdata_dir = "/home/lightcone/workspace/DRO-retarget/Noise-learn/data/dex_ycb"
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        # 遍历并加载所有的 meta.yml 文件

        self._prepare_data()
        self._load_camera_parameters()

    def _prepare_data(self):
        subject_dirs = sorted(
            Path(self._graspdata_dir).glob("*/"),
            key=lambda p: int(p.stem.split('-')[-1])
        )
        self._subject_dirs = []
        self._session_dirs = []

        idx1=0
        for subject_dir in subject_dirs:
            session_dirs = list(subject_dir.glob("*/"))
            self._subject_dirs.append(subject_dir.name)
            for session_dir in session_dirs:
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

                pose_m_flat = pose_m[:, 0, :]  # Flatten to (N, 51)
                mask = ~(pose_m_flat == 0).all(axis=1)  # Keep rows where not all zeros
                filtered_pose_m = pose_m[mask, :, :]  # Apply the mask to pose_m
                filtered_pose_y = pose_y[mask, :, :]  # Apply the mask to pose_y
                moving_objects_mask = (filtered_pose_y.max(axis=0) != filtered_pose_y.min(axis=0)).any(axis=1)
                filtered_pose_y = filtered_pose_y[:, moving_objects_mask, :] 

                self.metadata[subject_dir.name][session_dir.name]["mano_pose"] = filtered_pose_m
                self.metadata[subject_dir.name][session_dir.name]["obj_pose"] = filtered_pose_y


                moving_ycb_ids = [
                    self.metadata[subject_dir.name][session_dir.name]["ycb_ids"][i]
                    for i, is_moving in enumerate(moving_objects_mask) if is_moving
                ]

                for obj_id in moving_ycb_ids:
                    idx1+=1
                    self.object_names.append(YCB_CLASSES[obj_id])
                    obj_path = self._data_dir / "models" / YCB_CLASSES[obj_id] / "textured_simple.obj"
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
    def _load_camera_parameters(self):
        extrinsics = {}
        intrinsics = {}
        for cali_dir in self._calib_dir.iterdir():
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
            camera_mat = np.array([[x["fx"], 0.0, x["ppx"]], [0.0, x["fy"], x["ppy"]], [0.0, 0.0, 1.0]])
            intrinsics[name] = camera_mat
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def get_object_pc_and_mesh(self, object_name,obj_pose):
        random.seed(42)   # Setting Python's built-in random seed
        np.random.seed(42)  # Setting numpy's random seed
        object_mesh = self.object_meshes[object_name]
        sampled_object_pc = object_mesh.sample(5000)
        object_pc = torch.tensor(sampled_object_pc, dtype=torch.float)

        transformed_pc, transformed_mesh = apply_object_pose_transformation(object_pc, object_mesh, obj_pose)
        # transformed_pc*=1000/900
        # transformed_mesh.vertices*=1000/900
        return transformed_pc, transformed_mesh



    def __getitem__(self, index):
        if self.is_train:
            hand_name_batch = []
            object_name_batch = []
            object_pc_batch = []
            target_q_batch = []
            mano_mesh_batch = []
            mano_joint_batch = []
            wrist_quat_batch = []
            contact_point_indices_batch = []
            object_mesh_batch = []
            object_pose_batch = []
            dir_index_batch = []
            index+=0
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

                # extrinsic_mat = np.array(self.extrinsics[data['extrinsics']]["extrinsics"]["apriltag"]).reshape([3, 4])
                # extrinsic_mat = np.vstack([extrinsic_mat, [0, 0, 0, 1]])  # 转换为 4x4 齐次矩阵
                # pose_vec = pt.pq_from_transform(extrinsic_mat)
                # self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()

                start_frame = 10
                end_frame = min(hand_poses.shape[0], start_frame + 40)
                for i in range(start_frame,end_frame):
                    hand_pose = torch.tensor(hand_poses[i].squeeze(0))
                    # hand_pose*=0
                    
                    target_q_batch.append(hand_pose)

                    self.hand.set_parameters(hand_pose.unsqueeze(0),data_origin="ycb")

                    object_pose = data['obj_pose'][i].squeeze(0) 
                    # object_pose = self.camera_pose * sapien.Pose(object_pose[4:], np.concatenate([object_pose[3:4], object_pose[:3]]))

                    object_mesh = data[f"{object_name}_mesh"]
                    object_pc = object_mesh.sample(5000)  
                    object_pc = torch.tensor(object_pc, dtype=torch.float)
                    


                    transformed_pc, transformed_mesh = apply_object_pose_transformation(object_pc, object_mesh, object_pose)
                    transformed_pc*=1000/900
                    transformed_mesh.vertices*=1000/900
                    object_pc_batch.append(transformed_pc)
                    object_mesh_batch.append(transformed_mesh)

                    mano_mesh, mano_joint = self.hand.get_trans_trimesh_data(i=0, pose=None)
                    mano_mesh_batch.append(mano_mesh)
                    mano_joint=torch.tensor(mano_joint, dtype=torch.float32)
                    mano_joint_batch.append(mano_joint)

                    wrist_quat = hand_pose[:3] 
                    wrist_quat_batch.append(wrist_quat)

                    contact_point_indices=self.hand.get_contact_point_candidates(transformed_mesh)
                    contact_point_indices_batch.append(contact_point_indices)


                    object_pose[-3:]*=1000/900
                    object_pose_batch.append(object_pose)

                    # 可视化点云
                    # print(f"Hand pose: {hand_pose}")
                    # visualize_point_clouds(mano_mesh, transformed_pc)


            target_q_batch = torch.stack(target_q_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            mano_joint_batch = torch.stack(mano_joint_batch)  
            wrist_quat_batch = torch.stack(wrist_quat_batch)

            # object_pose_batch = torch.stack(object_pose_batch)
            import gc


            # gc.set_debug(gc.DEBUG_LEAK)
            unreachable_objects = gc.collect()
            print(f"Unreachable objects: {unreachable_objects}")
            return {
                "hand_pose": target_q_batch,
                "object_pcs": object_pc_batch,
                "contact_point_indices": contact_point_indices_batch,
                "mano_mesh": mano_mesh_batch,
                "mano_joint": mano_joint_batch,
                "wrist_quat": wrist_quat_batch,
                "object_meshes": object_mesh_batch,
                "object_pose": object_pose_batch,
                "object_name": object_name_batch,
                "dir_index": dir_index_batch
            }
        




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
            wrist_rot_batch = []
            object_mesh_batch = []
            object_pose_batch = []
            dir_index_batch = []

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


                start_frame = 0
                end_frame = min(hand_poses.shape[0], start_frame + 5)
                for i in range(start_frame,end_frame):
                    hand_pose = torch.tensor(hand_poses[i].squeeze(0)).to(self.device)
                    hand_pose*=0
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
                    wrist_rot = hand_pose[:3]  
                    wrist_rot_batch.append(wrist_rot)

                    translation = object_pose.p
                    quaternion = object_pose.q
                    whole_pose = np.concatenate([quaternion, translation])
                    whole_pose = torch.tensor(whole_pose, dtype=torch.float32)
                    object_pose_batch.append(whole_pose)


            target_q_batch = torch.stack(target_q_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            mano_joint_batch = torch.stack(mano_joint_batch) 
            print("mano_joint_batch",mano_joint_batch) 
            wrist_rot_batch = torch.stack(wrist_rot_batch)
            import gc
            unreachable_objects = gc.collect()
            return {
                "hand_pose": target_q_batch,
                "object_pcs": object_pc_batch,
                "mano_mesh": mano_mesh_batch,
                "mano_joint": mano_joint_batch,
                "wrist_rot": wrist_rot_batch,
                "object_meshes": object_mesh_batch,
                "object_pose": object_pose_batch,
                "object_name": object_name_batch,
                "dir_index": dir_index_batch
            }
    


