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
from utils.object_model import ObjectModel
from utils.hand_model import create_hand_model
from transforms3d.euler import euler2mat
import plotly.graph_objects as go
import torch
from pytransform3d import rotations
def plane2pose(plane_parameters):
    r3 = plane_parameters[:3]
    r2 = torch.zeros_like(r3)
    r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0) if r3[2] * r3[2] <= 0.5 else (-r3[2], 0, r3[0])
    r1 = torch.cross(r2, r3)
    pose = torch.zeros([4, 4], dtype=torch.float, device=plane_parameters.device)
    pose[0, :3] = r1
    pose[1, :3] = r2
    pose[2, :3] = r3
    pose[2, 3] = plane_parameters[3]
    pose[3, 3] = 1
    return pose


def visualize_transformed_links_and_object_pc(robot_pc, object_pc=None):
    """
    Visualize the transformed robot links' point cloud and object point cloud using Plotly.
    Additionally, highlight the extreme points (up, down, left, right) for each robot link.

    Args:
        robot_pc (torch.Tensor): Robot point cloud with shape (N, 4),
        where the last column is the link index.
        object_pc (torch.Tensor, optional): Object point cloud with shape (M, 3).
    """
    import plotly.graph_objects as go
    import numpy as np

    # Convert robot point cloud to numpy
    robot_pc = robot_pc.detach().cpu().numpy()

    # Extract robot coordinates and link indices
    robot_x, robot_y, robot_z = robot_pc[:, 0], robot_pc[:, 1], robot_pc[:, 2]
    robot_link_indices = robot_pc[:, 3]

    # Create a Plotly figure
    fig = go.Figure()

    # Plot each robot link's point cloud
    unique_links = sorted(set(robot_link_indices))
    robot_colors = [
        'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'pink', 'yellow'
    ]  # Color palette for robot links

    for i, link_index in enumerate(unique_links):
        link_mask = (robot_link_indices == link_index)  # Mask for the current link
        link_x = robot_x[link_mask]
        link_y = robot_y[link_mask]
        link_z = robot_z[link_mask]

        # Find extreme points
        extreme_points = {
            'x_max': np.argmax(link_x),
            'x_min': np.argmin(link_x),
            'y_max': np.argmax(link_y),
            'y_min': np.argmin(link_y),
            'z_max': np.argmax(link_z),
            'z_min': np.argmin(link_z),
        }
        extreme_coords = {
            key: [link_x[idx], link_y[idx], link_z[idx]]
            for key, idx in extreme_points.items()
        }

        # Add link point cloud to the plot
        fig.add_trace(go.Scatter3d(
            x=link_x,
            y=link_y,
            z=link_z,
            mode='markers',
            marker=dict(size=2, color=robot_colors[i % len(robot_colors)]),
            name=f'Robot Link {int(link_index)}'
        ))

        # Add extreme points to the plot
        for key, coord in extreme_coords.items():
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=5, color='black', symbol='diamond'),
                name=f'Link {int(link_index)} {key}'
            ))


    # Plot object point cloud (if provided)
    if object_pc is not None:
        # Convert object point cloud to numpy
        object_pc = object_pc.detach().cpu().numpy()
        object_x, object_y, object_z = object_pc[:, 0], object_pc[:, 1], object_pc[:, 2]

        # Add object point cloud as a separate trace
        fig.add_trace(go.Scatter3d(
            x=object_x,
            y=object_y,
            z=object_z,
            mode='markers',
            marker=dict(size=3, color='black'),
            name='Object Point Cloud'
        ))

    # Configure the layout
    fig.update_layout(
        title="Robot and Object Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Show the figure
    fig.show()


class ManoDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        hand_name: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 5000,
        object_pc_type: str = 'random',
        noise_level: str = 'pointcloud',
        mano_root:str = None
    ):
        self.noise_level = noise_level
        self.batch_size = batch_size
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        device = 'cpu'

        self.hand_name="mano"
        self.hand=create_hand_model(self.hand_name, torch.device('cpu'))
        
        
        self._data_dir = Path(mano_root)
        print(self._data_dir,"aaa")
        self._calib_dir = self._data_dir / "calibration"
        self._model_dir = self._data_dir / "models"
        self._graspdata_dir = self._data_dir / "graspdata"
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        split_json_path =  Path('/home/lightcone/workspace/DRO-retarget/Noise-learn/thirdparty/manopth/mano_meshdata.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'][:100] if is_train else dataset_split['validate']
        
        for object_name in self.object_names:
            data_dicts = np.load(os.path.join(self._graspdata_dir, object_name + '.npy'), allow_pickle=True)
            self.metadata[self.hand_name][object_name] = {}
            
            for num, data_entry in enumerate(data_dicts):
                scale = data_entry["scale"]
                plane = data_entry["plane"]
                qpos = data_entry["qpos"]
                contact_point_indices = data_entry["contact_point_indices"]
                qpos_st = data_entry["qpos_st"]

                # 直接转换欧拉角为旋转矩阵，并提取前两列（简洁写法）
                rot_matrix = euler2mat(*qpos['rot'])[:, :2].T.ravel()
                st_rot_matrix = euler2mat(*qpos_st['rot'])[:, :2].T.ravel()
                # 构造 hand_pose，直接拼接列表
                hand_pose = torch.concat([torch.tensor(qpos[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])
                hand_pose_st = torch.concat([torch.tensor(qpos_st[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])

                # 简化 metadata 字典的赋值过程
                self.metadata[self.hand_name][object_name][num] = {
                    "scale": scale,
                    "plane": plane,
                    "handpos": hand_pose,
                    "contact_point_indices": contact_point_indices,
                    "qpos_st": hand_pose_st,
                }
                
        self.object_pcs = {}
        if self.object_pc_type != 'fixed':
            self.object_model = ObjectModel(
                data_root_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/meshdata',
                batch_size_each=1,
                num_samples=0, 
                device=device
            )
            self.object_model.initialize(self.object_names)


            scales = [
                [
                    data['scale']
                    for data in self.metadata[self.hand_name][object_name].values()
                ]
                for object_name in self.object_names
            ]
            max_len = max(len(sublist) for sublist in scales)

            # 使用填充（例如填充0.0）使所有子列表长度一致
            padded_scales = [
                sublist + [0.0] * (max_len - len(sublist))  # 用0填充到最大长度
                for sublist in scales
            ]

            # 转换为 PyTorch 张量
            object_scale = torch.tensor(
                padded_scales, 
                dtype=torch.float, 
                device=device
            )
            self.object_model.object_scale_tensor = object_scale

            for i, object_name in enumerate(self.object_names):
                mesh = self.object_model.get_mesh(i)
                object_pc, _ = mesh.sample(5000, return_index=True)
                self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
        else:
            print("!!! Using fixed object pcs !!!")

            
            
            
    def __getitem__(self, index):
        if self.is_train:
            hand_name_batch = []
            object_name_batch = []
            robot_links_pc_batch = []
            robot_pc_initial_batch = []
            robot_pc_target_batch = []
            object_pc_batch = []
            dro_gt_batch = []
            random_q_batch = []
            target_q_batch = []
            mano_mesh_batch = []
            mano_joint_batch = []
            wrist_quat_batch = []
            wrist_pos_batch = []
            contact_point_indices_batch = []
            object_mesh_batch = []

            for idx in range(self.batch_size):
                hand_name_batch.append(self.hand_name)
                object_index = (index * self.batch_size + idx) % len(self.object_names)
                # object_index = random.choice([0,1])
                object_name = self.object_names[object_index]
                object_name_batch.append(object_name)
                # print(object_name)
                hand = self.hand
                for num, data in self.metadata[self.hand_name][object_name].items():
                    # if num>0:
                    #     continue
                    # print(f"Processing metadata entry for object: {object_name}, num: {num}")
                    # print(f"Data: {data}")
                    plane=torch.tensor(data['plane'], dtype=torch.float)  # plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                    # pose = plane2pose(plane)
                    if 'qpos_st' in data:
                        qpos_st = data['qpos_st'].clone().detach() if isinstance(data['qpos_st'], torch.Tensor) else torch.tensor(data['qpos_st'], dtype=torch.float32)
                        qpos_st = qpos_st.unsqueeze(0)
                        # hand.set_parameters(qpos_st)
                        # hand_st_plotly = hand.get_plotly_data(i=0, opacity=0.5, color='lightblue', pose=pose)
                    
                    indices = torch.randperm(5000)[:self.num_points]
                    object_pc = self.object_pcs[object_name][indices]
                    object_mesh = self.object_model.get_mesh(object_index)
                    model_scale = self.object_model.object_scale_tensor[object_index, num].detach().cpu().numpy()
                    # object_plotly = self.object_model.get_plotly_data(i=object_index, model_scale=model_scale,  color='lightgreen', opacity=1, pose=pose)
                    object_pc = object_pc * model_scale *1000/900
                    object_mesh.vertices = object_mesh.vertices * model_scale *1000/900
                    # if pose is not None:
                    #     pose = np.array(pose, dtype=np.float32)
                    #     object_pc = object_pc @ pose[:3, :3].T + pose[:3, 3]
                    object_pc_batch.append(object_pc)
                    object_mesh_batch.append(object_mesh)
                    
                    if 'contact_point_indices' in data:
                        hand_pose = data['handpos'].clone().detach() if isinstance(data['handpos'], torch.Tensor) else torch.tensor(data['handpos'], dtype=torch.float32)
                        print(hand_pose.shape,"handpos"*100)
                        # hand_pose *=0
                        contact_point_indices = data['contact_point_indices'].clone().detach() if isinstance(data['contact_point_indices'], torch.Tensor) else torch.tensor(data['contact_point_indices'], dtype=torch.long)
                        # print(hand_pose.unsqueeze(0).shape, contact_point_indices.shape)
                       
                        hand.set_parameters(hand_pose.unsqueeze(0), contact_point_indices.unsqueeze(0))
                        contact_point_indices=self.hand.get_contact_point_candidates(object_mesh)
                        # hand_en_plotly = hand.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=True, pose=pose)
                        wrist_pos = hand_pose[:3]
                        wrist_quat = hand_pose[3:6] 
                        # combined_rot_matrix=rotations.matrix_from_quaternion(wrist_quat) @ pose[:3, :3]^T
                        # wrist_quat = pr.quaternion_from_matrix(combined_rot_matrix)
                        
                        wrist_quat = torch.tensor(wrist_quat, dtype=torch.float32)
                        wrist_pos_batch.append(wrist_pos)
                        wrist_quat_batch.append(wrist_quat)
                        # root_translation = hand_pose[:3]  # 提取平移部分
                        # root_rotation = hand_pose[3:6]    # 提取旋转部分
                        # # 应用 pose 变换
                        # if pose is not None:
                        #     # 确保 pose 是 numpy 格式
                        #     # if isinstance(pose, torch.Tensor):
                        #     #     pose_numpy = pose.detach().cpu().numpy()
                        #     # pose_numpy = np.array(pose, dtype=np.float32)

                        #     # 平移变换
                        #     root_translation = pose[:3, :3] @ root_translation + pose[:3, 3]

                        #     # 旋转变换
                        #     # 注意：需要根据 root_rotation 的具体格式处理
                        #     root_rotation =  pose[:3, :3] @ root_rotation

                        # # 更新 hand_pose 的根部 6 位
                        # hand_pose[:3] = root_translation
                        # hand_pose[3:6] = root_rotation
                        target_q_batch.append(hand_pose)
                        contact_point_indices_batch.append(contact_point_indices)
                        mano_mesh,mano_joint = hand.get_trans_trimesh_data(i=0,pose=None)
                        mano_mesh_batch.append(mano_mesh)
                        mano_joint=torch.tensor(mano_joint, dtype=torch.float32)
                        mano_joint_batch.append(mano_joint)



                    # if num==10000:
                    #     fig = go.Figure(hand_en_plotly+ object_plotly)
                    #     fig.update_layout(scene_aspectmode='data')
                    #     fig.show()

            target_q_batch = torch.stack(target_q_batch)
            # contact_point_indices_batch = torch.stack(contact_point_indices_batch) 
            object_pc_batch = torch.stack(object_pc_batch)
            mano_joint_batch = torch.stack(mano_joint_batch)  
            wrist_quat_batch = torch.stack(wrist_quat_batch)
            return {
                "hand_pose": target_q_batch,
                "contact_point_indices": contact_point_indices_batch,
                "object_pcs": object_pc_batch,
                "mano_mesh": mano_mesh_batch,
                "mano_joint": mano_joint_batch,
                "wrist_quat": wrist_quat_batch,
                "wrist_pos": wrist_pos_batch,
                "object_meshes": object_mesh_batch,
            }
                