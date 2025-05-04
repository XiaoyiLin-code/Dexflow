from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
import torch
from pytransform3d import rotations

from retarget_utils.robot_wrapper import RobotWrapper
from retarget_utils.constants import RobotName, HandType, OPERATOR2MANO, SCALE
from retarget_utils.optimizer import PositionOptimizer
from config.path_config import Path_config
from utils.hand_model import create_hand_model
from optimize_utils.finger_optimizer import FingerOptimizer
from other_utils.visualize import *  

class RetargetWrapper:
    """
    A wrapper class to handle the initialization and retargeting of robot hands.
    It utilizes various optimizers and models to retarget the robot's hand to match
    the human hand model.
    """
    
    DUMMY_JOINT_NAMES = [
        "dummy_x_translation_joint", "dummy_y_translation_joint", "dummy_z_translation_joint",
        "dummy_x_rotation_joint", "dummy_y_rotation_joint", "dummy_z_rotation_joint"
    ]
    
    def __init__(self, robot_names: List[str], hand_type: HandType, dataset, cfgs):
        """
        Initializes the RetargetWrapper class with robot names, hand type, dataset, and configurations.

        Args:
            robot_names (List[str]): A list of robot names.
            hand_type (HandType): Type of hand (e.g., left or right).
            dataset (Dataset): The dataset containing hand and wrist data.
            cfgs (dict): Configuration settings for retargeting and optimization.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot_names = robot_names
        self.hand_type = hand_type
        self.robot_wrappers: Dict[RobotWrapper] = {}
        self.Optimizers: Dict[PositionOptimizer] = {}
        self.data_path = Path(Path_config.get_data_path())
        self.dataset = dataset
        self.cfgs = cfgs

        self.hand_models = {robot_name.name:
            create_hand_model(robot_name.name, self.hand_type.name, torch.device("cpu")) 
            for robot_name in self.robot_names
        }
        self._initialize_robots()
        self.retarget_qpos = {}

    def _initialize_robots(self):
        """
        Initializes the robot wrappers and optimizers based on the given configuration.
        """
        robot_file_paths = [
            str(self.data_path / f"urdf/dex_retargeting/hands/{robot_name.name}_hand/{robot_name.name}_hand_{self.hand_type.name}_glb_w_dummy.urdf") 
            for robot_name in self.robot_names
        ]
        
        self.robot_wrappers = {robot_name.name: RobotWrapper(robot_file_path) for robot_name,robot_file_path in zip(self.robot_names,robot_file_paths)}
        
        self.Optimizers = {robot_name.name:
            PositionOptimizer(
                self.robot_wrappers[robot_name.name],
                self.cfgs[i]["retargeting"]["target_joint_names"],
                self.cfgs[i]["retargeting"]["target_link_names"],
                self.cfgs[i]["retargeting"]["target_link_human_indices"]
            ) 
            for i,robot_name in enumerate(self.robot_names)
        }
        
        for opt, wrapper in zip(self.Optimizers.values(), self.robot_wrappers.values()):
            opt.set_joint_limit(wrapper.joint_limits[opt.idx_pin2target])
        
        self.Finger_Optimizers = [
            FingerOptimizer(
                hand_name=robot_name,
                w_dis=500,
                w_pen=1e9,
                w_joints=1e-2
            ) 
            for robot_name in self.robot_names
        ]

    def warm_start(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray, hand_type: HandType = HandType.right, 
                   robot: RobotWrapper = None, optimizer=None, qpos=None):
        """
        Performs a warm start to initialize the robot's wrist position and orientation.
        
        Args:
            wrist_pos (np.ndarray): The wrist position.
            wrist_quat (np.ndarray): The wrist orientation in quaternion form.
            hand_type (HandType): Type of hand (left or right).
            robot (RobotWrapper): The robot wrapper to be used.
            optimizer (PositionOptimizer): The optimizer for position.
        """
        target_wrist_pose = np.eye(4)

        # target_wrist_pose[:3, :3] = rotations.matrix_from_eular(wrist_quat)
        target_wrist_pose[:3, 3] = wrist_pos
        
        # wrist_link_id = robot.get_joint_parent_child_frames(self.DUMMY_JOINT_NAMES[5])[1]
        # old_qpos = robot.q0
        # new_qpos = old_qpos.copy()
        
        # # Set joints that belong to dummy joints to 0
        # new_qpos[[i for i, joint_name in enumerate(optimizer.target_joint_names) if joint_name in self.DUMMY_JOINT_NAMES]] = 0
        
        # robot.compute_forward_kinematics(new_qpos)
        # root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose
        # euler = rotations.euler_from_matrix(target_root_pose[:3, :3], 0, 1, 2, extrinsic=False)
        pose_vec = np.concatenate([target_root_pose[:3, 3], wrist_quat])
        
        for num, joint_name in enumerate(optimizer.target_joint_names):
            if joint_name in self.DUMMY_JOINT_NAMES:
                qpos[num] = pose_vec[self.DUMMY_JOINT_NAMES.index(joint_name)]

        self.is_warm_started = True
        return qpos

    def retarget(self,data,robot_name):
        DEBUG = False
        """
        Retargets the robot's hand model based on the dataset, adjusting positions 
        and orientations according to human hand data.

        Returns:
            List: A list of retargeted joint positions.
        """
        retargeted_qpos = []
    
        mano_joint = data["mano_joint"]
        wrist_quats = data["wrist_quat"]
        object_meshes = data["object_meshes"]
        object_pcs = data["object_pcs"]
        optimizer = self.Optimizers[robot_name.name]
        robot_wrapper = self.robot_wrappers[robot_name.name]
        scale = SCALE[robot_name.name]
        indices = optimizer.target_link_human_indices
        
        for i, (joint, quat) in enumerate(zip(mano_joint, wrist_quats)):

            object_mesh = object_meshes[i]
            object_mesh.vertices *= 1000 / scale
            joint = joint.cpu().numpy() * 1000 / scale

            mano_mesh = data["mano_mesh"][i]
            mano_mesh.vertices *= 1000 / scale

            init_qpos = torch.tensor(robot_wrapper.joint_limits.mean(1)[optimizer.idx_pin2target].astype(np.float32))
            init_qpos[3:6]=torch.tensor(-quat.as_euler('xyz', degrees=False))
            init_qpos[:3] = torch.tensor(joint[0, :3])
            init_qpos = self.warm_start(joint[0, :], -quat.as_euler('xyz', degrees=False), hand_type=self.hand_type, robot=robot_wrapper, optimizer=optimizer,qpos=init_qpos)
            ref_value = joint[indices, :]
            fixed_qpos = np.array([])
            robot_pc=self.hand_models[robot_name.name].get_transformed_links_gene_pc(
                            torch.tensor(init_qpos, dtype=torch.float32).to(self.device)

                    )
            retarget_qpos = optimizer.retarget(ref_value, fixed_qpos, init_qpos.cpu().numpy())
            retargeted_qpos.append(retarget_qpos)
            if DEBUG:
                object_pc = object_pcs[i]
                object_pc *= 1000 / scale
                visualize_and_save_transformed_links_and_object_pc(
                        robot_pc=robot_pc,
                        mano_mesh=mano_mesh,
                    )
    
        return retargeted_qpos

