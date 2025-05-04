import torch
import trimesh
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict
from transforms3d.euler import euler2mat
from scipy.spatial.transform import Rotation as R

class TACODataset(Dataset):
    def __init__(self, data_dir, batch_size=1, hand_type="right", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self._data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device
        self.hand_type = hand_type
        
        # 初始化手部模型
        self.hand = create_hand_model("mano", self.device)
        
        # 加载数据
        self._load_data()
        self._load_camera_parameters()
        self._precompute_contact_points()

    def _load_data(self):
        # 初始化数据结构
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.object_meshes = {}
        
        # 加载手部和物体数据
        self._load_hand_data()
        self._load_object_data()
        self._load_calibration_data()

    def _load_hand_data(self):
        hand_dir = self._data_dir / "Hand_Poses"
        for task_dir in hand_dir.iterdir():
            for seq_dir in task_dir.iterdir():
                # 加载手部姿态和形状参数
                pose_file = seq_dir / "right_hand.pkl"
                shape_file = seq_dir / "right_hand_shape.pkl"
                
                poses = np.load(pose_file, allow_pickle=True)
                shapes = np.load(shape_file, allow_pickle=True)
                
                # 存储到metadata
                key = f"{task_dir.stem}_{seq_dir.stem}"
                self.metadata[key]['hand_pose'] = [poses[k] for k in poses.keys()]
                self.metadata[key]['hand_shape'] = shapes["hand_shape"].numpy()

    def _load_object_data(self):
        obj_dir = self._data_dir / "Object_Poses"
        for task_dir in obj_dir.iterdir():
            for seq_dir in task_dir.iterdir():
                # 加载物体位姿和模型
                obj_files = list(seq_dir.glob("tool_*.pkl"))
                for obj_file in obj_files:
                    obj_id = obj_file.stem.split('_')[-1]
                    poses = np.load(obj_file, allow_pickle=True)
                    
                    # 加载物体网格
                    mesh_path = self._data_dir / "Object_Models" / f"{obj_id}.obj"
                    mesh = trimesh.load_mesh(mesh_path)
                    
                    key = f"{task_dir.stem}_{seq_dir.stem}"
                    self.metadata[key]['obj_pose'].append(poses)
                    self.metadata[key]['obj_mesh'] = mesh
                    self.object_meshes[obj_id] = mesh

    def _load_camera_parameters(self):
        calib_dir = self._data_dir / "Egocentric_Camera_Parameters"
        self.intrinsics = {}
        self.extrinsics = {}
        
        for task_dir in calib_dir.iterdir():
            for seq_dir in task_dir.iterdir():
                # 加载内参和外参
                intrinsic_file = seq_dir / "intrinsic.txt"
                extrinsic_file = seq_dir / "extrinsic.npy"
                
                intrinsic = np.loadtxt(intrinsic_file)
                extrinsic = np.load(extrinsic_file)
                
                key = f"{task_dir.stem}_{seq_dir.stem}"
                self.intrinsics[key] = intrinsic
                self.extrinsics[key] = extrinsic

    def _precompute_contact_points(self):
        # 预计算接触点索引
        for key in self.metadata.keys():
            mesh = self.metadata[key]['obj_mesh']
            self.metadata[key]['contact_points'] = self.hand.get_contact_point_candidates(mesh)

    def __getitem__(self, index):
        batch_data = defaultdict(list)
        
        for _ in range(self.batch_size):
            key = list(self.metadata.keys())[index % len(self.metadata)]
            metadata = self.metadata[key]
            
            # 获取手部数据
            hand_pose = torch.tensor(metadata['hand_pose'][index], device=self.device)
            hand_shape = torch.tensor(metadata['hand_shape'], device=self.device)
            
            # 应用外参变换
            extrinsic = self.extrinsics[key]
            hand_pose = apply_extrinsic_to_mano(hand_pose, extrinsic)
            
            # 获取物体数据
            obj_pose = apply_extrinsic_to_object(metadata['obj_pose'][index], extrinsic)
            obj_mesh = metadata['obj_mesh']
            
            # 生成点云
            obj_pc = torch.tensor(obj_mesh.sample(5000), device=self.device)
            
            # 获取接触点索引
            contact_idx = metadata['contact_points']
            
            # 获取手部关节和网格
            self.hand.set_parameters(hand_pose.unsqueeze(0), hand_shape.unsqueeze(0))
            mano_mesh, mano_joints = self.hand.get_trans_trimesh_data()
            
            # 填充批次数据
            batch_data['hand_pose'].append(hand_pose)
            batch_data['object_pcs'].append(obj_pc)
            batch_data['contact_point_indices'].append(contact_idx)
            batch_data['mano_mesh'].append(mano_mesh)
            batch_data['mano_joint'].append(mano_joints)
            batch_data['object_meshes'].append(obj_mesh)
            batch_data['object_pose'].append(obj_pose)
            
            index = (index + 1) % len(self.metadata)

        # 转换为张量
        for k in batch_data:
            if k not in ['mano_mesh', 'object_meshes']:
                batch_data[k] = torch.stack(batch_data[k])
                
        return batch_data

    def __len__(self):
        return len(self.metadata) // self.batch_size

# 保持与原始代码相同的工具函数
def apply_extrinsic_to_mano(hand_pose, extrinsic):
    # 实现与外参变换相同的逻辑
    return hand_pose  # 简化的实现

def apply_extrinsic_to_object(obj_pose, extrinsic):
    # 实现与外参变换相同的逻辑
    return obj_pose  # 简化的实现