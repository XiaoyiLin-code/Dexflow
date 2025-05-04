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
def visualize_transformed_links_and_object_pc(robot_pc, object_pc=None):
    """
    Visualize the transformed robot links' point cloud and object point cloud using Plotly.

    Args:
        robot_pc (torch.Tensor): Robot point cloud with shape (N, 4),
        where the last column is the link index.
        object_pc (torch.Tensor, optional): Object point cloud with shape (M, 3).
    """
    import plotly.graph_objects as go

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
        fig.add_trace(go.Scatter3d(
            x=robot_x[link_mask],
            y=robot_y[link_mask],
            z=robot_z[link_mask],
            mode='markers',
            marker=dict(size=2, color=robot_colors[i % len(robot_colors)]),
            name=f'Robot Link {int(link_index)}'
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


class TrainDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_name: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random',
        noise_level: str = 'pointcloud'
    ):
        self.noise_level = noise_level
        self.batch_size = batch_size
        self.robot_name = robot_name if robot_name is not None \
            else ['shadowhand']
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        device = 'cpu'


        self.hand=create_hand_model(self.robot_name, torch.device('cpu'))
        self.dofs=len(self.hand.pk_chain.get_joint_parameter_names())
        
        split_json_path =  Path('/home/lightcone/workspace/DRO-retarget/Noise-learn/data_utils/meshdata.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'][:100] if is_train else dataset_split['validate']
        if debug_object_names is not None:
            print("!!! Using debug objects !!!")
            self.object_names = debug_object_names
        joint_data_path=Path('/home/lightcone/workspace/DRO-retarget/Noise-learn/data/dexgraspnet')
        translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
        rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
        joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
        for object_name in self.object_names:
            data_dicts = np.load(os.path.join(joint_data_path, object_name + '.npy'), allow_pickle=True)
            self.metadata[self.robot_name][object_name] = {}

            for num in range(len(data_dicts)):
                data_entry = data_dicts[num]
                print(data_entry.keys())
                # 提取 qpos 数据
                qpos = data_entry['qpos']
                
                # # 提取旋转信息并转换为矩阵
                rot_matrix = np.array(euler2mat(*[qpos[name] for name in rot_names]))
                rot = rot_matrix[:, :2].T.ravel().tolist()  # 提取前两列并展平

                # 构造 hand_pose
                hand_pose = [qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names]
                # print(hand_pose,qpos)
                # 存储到 metadata
                self.metadata[self.robot_name][object_name][num] = {
                    'qpos': np.array(hand_pose),  # 将 hand_pose 存为 NumPy 数组
                    'scale': data_entry['scale']
                }
                
        if not self.is_train:
            self.combination = []
            for object_name in self.object_names:
                self.combination.append((self.robot_name, object_name))
            self.combination = sorted(self.combination)
        
        self.object_pcs = {}
        if self.object_pc_type != 'fixed':
            self.object_model = ObjectModel(
                data_root_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/meshdata',
                batch_size_each=1,
                num_samples=0, 
                device=device
            )
            self.object_model.initialize(self.object_names)
            print(self.metadata[self.robot_name][object_name])

            scales = [
                [
                    data['scale']
                    for data in self.metadata[self.robot_name][object_name].values()
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
            robot_name_batch = []
            object_name_batch = []
            robot_links_pc_batch = []
            robot_pc_initial_batch = []
            robot_pc_target_batch = []
            object_pc_batch = []
            dro_gt_batch = []
            random_q_batch = []
            target_q_batch = []
            for idx in range(self.batch_size):
                robot_name_batch.append(self.robot_name)
                hand = self.hand
                #index = random.randint(0, len(self.object_names) - 1)
                index=0
                object_name=self.object_names[index]
                #number= random.randint(0, len(self.metadata[self.robot_name][object_name]) - 1)
                number=0
                print(object_name,index)

                target_q = self.metadata[self.robot_name][object_name][number]['qpos']
                target_q_batch.append(target_q)
                object_name_batch.append(object_name)

                robot_links_pc_batch.append(hand.links_pc)
                if self.object_pc_type == 'fixed':
                    name = object_name.split('+')
                    object_path = Path('/home/lightcone/workspace/DRO-retarget/DRO-Grasp/data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]
                elif self.object_pc_type == 'random':
                    indices = torch.randperm(5000)[:self.num_points]
                    object_pc = self.object_pcs[object_name][indices]
                    model_index = index // self.object_model.batch_size_each

                    model_scale = self.object_model.object_scale_tensor[model_index, number].detach().cpu().numpy()
                    object_pc = object_pc * model_scale
                    # object_pc += torch.randn(object_pc.shape) * 0.002
                else:  # 'partial', remove 50% points
                    indices = torch.randperm(5000)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_name][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]

                object_pc_batch.append(object_pc)
                # print(target_q.shape,target_q)

                if not isinstance(target_q, torch.Tensor):
                    target_q = torch.tensor(target_q, dtype=torch.float32)
                # print(self.hand.dof_name)

                robot_pc_target = hand.get_transformed_links_pc(target_q)
                # visualize_transformed_links_and_object_pc(robot_pc_target, object_pc)
                robot_pc_target=robot_pc_target
                robot_pc_target_batch.append(robot_pc_target)
                # if self.noise_level == "joint":
                #     random_q = hand.get_random_q(target_q)
                # elif self.noise_level == "pointcloud":
                random_q = hand.get_random_q(target_q)
                random_q_batch.append(random_q)
                robot_pc_initial = hand.get_transformed_links_pc(random_q)
                robot_pc_initial_batch.append(robot_pc_initial)

                # dro = torch.cdist(robot_pc_target, object_pc, p=2)
                # dro_gt_batch.append(dro)

            # robot_pc_initial_batch = torch.stack(robot_pc_initial_batch)
            robot_pc_target_batch = torch.stack(robot_pc_target_batch)
            object_pc_batch = torch.stack(object_pc_batch)
            # dro_gt_batch = torch.stack(dro_gt_batch)

            B, N = self.batch_size, self.num_points
            # assert robot_pc_initial_batch.shape == (B, N, 3),\
            #     f"Expected: {(B, N, 3)}, Actual: {robot_pc_initial_batch.shape}"
            assert robot_pc_target_batch.shape == (B, N, 4),\
                f"Expected: {(B, N, 3)}, Actual: {robot_pc_target_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3),\
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
            # assert dro_gt_batch.shape == (B, N, N),\
            #     f"Expected: {(B, N, N)}, Actual: {dro_gt_batch.shape}"

            return {
                'robot_name': robot_name_batch,  # list(len = B): str
                'object_name': object_name_batch,  # list(len = B): str
                'robot_links_pc': robot_links_pc_batch,  # list(len = B): dict, {link_name: (N_link, 3)}
                'robot_pc_initial': robot_pc_initial_batch,
                'robot_pc_target': robot_pc_target_batch,
                'object_pc': object_pc_batch,
                # 'dro_gt': dro_gt_batch,
                'random_q': random_q_batch,
                'target_q': target_q_batch
            }

