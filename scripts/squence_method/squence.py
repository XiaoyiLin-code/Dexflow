from scripts.retarget.retarget_wrapper import RetargetWrapper
from optimize_utils.finger_optimizer import FingerOptimizer
from typing import List, Dict
from retarget_utils.constants import RobotName, HandType,OPERATOR2MANO,SCALE
from config.path_config import Path_config
from pathlib import Path
import yaml
import numpy as np
import torch

def load_yaml( path: Path):
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, "r") as file:
        return yaml.safe_load(file)

class SquenceRefine_Wrapper:
    def __init__(self,robot_names: List[RobotName],hand_models):
        self.robot_names = robot_names
        self.hand_models = hand_models
        self.optimized_hand_poses = []
        self.Finger_Optimizers = {robot_name.name: FingerOptimizer(
            hand_name=robot_name.name,
            w_dis=500,
            w_pen=1e11,
            w_joints=1e-2,
        ) for robot_name in self.robot_names}
        self.base_dir = Path(Path_config.BASE_DIR)
        self.load_finger_config()

    def init_pose_selector(self,init_qpos):
        #TODO: add a function to select the optimizer init pose in order to avoid error caused by original pose data
        return init_qpos
    
    def load_finger_config(self):
        config_paths = [self.base_dir / f"config/{robot_name.name}_finger_indices.yml" for robot_name in self.robot_names]
        configs = [load_yaml(config_path) for config_path in config_paths]
        self.finger_joints_indices = {robot_name.name: config["finger_joints_indices"] for robot_name,config in zip(self.robot_names,configs)}

    def optimize_finger(self, robot_name, init_qpos, object_mesh, object_pcs, indices_dict, robot_indices_dict):
        self.init_qpos = self.init_pose_selector(init_qpos)
        optimizer = self.Finger_Optimizers[robot_name.name]
        hand_model = self.hand_models[robot_name.name]
        for qpos, contact_points_per_finger, object_contact_point_indices,mesh,object_pc in zip(self.init_qpos,robot_indices_dict,indices_dict,object_mesh,object_pcs):
            qpos = torch.tensor(qpos, dtype=torch.float32)
            optimizer.set_up(
                hand_model=hand_model,
                object_surface=mesh,
                contact_points_per_finger=contact_points_per_finger,
                finger_joints_indices=self.finger_joints_indices[robot_name.name],
                initial_hand_pose=qpos,
                object_contact_point_indices = object_contact_point_indices,
                object_pcs=object_pc,
            )
            optimized_hand_pose, finger_scores = optimizer.optimize_all_fingers()       
            self.optimized_hand_poses.append(optimized_hand_pose)
            return self.optimized_hand_poses
