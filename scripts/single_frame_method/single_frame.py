from typing import List, Dict
import numpy as np
from retarget_utils.robot_wrapper import RobotWrapper
from enum import Enum
from retarget_utils.constants import RobotName, HandType,OPERATOR2MANO,SCALE
from retarget_utils.optimizer import PositionOptimizer
import yaml
from pytransform3d import rotations
from pathlib import Path
import time
import re
import os
import sys
from collections.abc import Mapping, Iterable
from config.path_config import Path_config

from data_utils.retarget_data import ManoDataset
from data_utils.retarget_data_dexycb import DexYCB_Dataset
from utils.hand_model import create_hand_model
import torch
from optimize_utils.finger_optimizer import FingerOptimizer
from kaolin.metrics.pointcloud import sided_distance
import json

from other_utils.visualize import *

from pathlib import Path
import torch
import yaml
from typing import List
import trimesh

from scripts.contact_mapping.contact_mapping import ContactMapping

class RetargetWrapper:
    def __init__(self, robot_names: List[RobotName], hand_type: HandType):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot_names = robot_names
        self.hand_type = hand_type
        self.robot_wrappers: List[RobotWrapper] = []
        self.Optimizers: List[PositionOptimizer] = []
        self.Optimizers2: List[FingerOptimizer] = []
        self.base_dir = Path(Path_config.BASE_DIR)
        self.third_party_path = Path(Path_config.get_third_party_path())
        self.data_path = Path(Path_config.get_data_path())

        self.cfgs = [self.load_yaml((
            self.base_dir 
            / "config" 
            / "mapping_config"
            / f"{robot_name.name}_hand_{self.hand_type.name}.yml"
        )) for robot_name in self.robot_names]
        self.dataset = DexYCB_Dataset(
            batch_size=1,
            select_idx=[0],
        )
        self.hand_models = [create_hand_model(robot_name.name,self.hand_type.name, torch.device("cpu")) for robot_name in self.robot_names]
        self.load_finger_config()
        self.initialize_robots()
        self.contact_mapping = ContactMapping(upper_bound=0.0007,
                                    lower_bound=0.0003,
                                    device=self.device)
     
    def load_yaml(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"配置文件未找到: {path}")
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def load_finger_config(self):
        config_paths = [self.base_dir / f"config/{robot_name.name}_finger_indices.yml" for robot_name in self.robot_names]
        configs = [self.load_yaml(config_path) for config_path in config_paths]
        self.finger_joints_indices = {robot_name.name:config["finger_joints_indices"] for robot_name,config in zip(self.robot_names,configs)}


    def initialize_robots(self):
        robot_file_paths = [str(self.data_path / f"urdf/dex_retargeting/hands/{robot_name.name}_hand/{robot_name.name}_hand_{self.hand_type.name}_glb_w_dummy.urdf") for robot_name in self.robot_names]
        self.robot_wrappers = [RobotWrapper(robot_file_path) for robot_file_path in robot_file_paths]
        self.Optimizers = [PositionOptimizer(
            self.robot_wrappers[i],
            self.cfgs[i]["retargeting"]["target_joint_names"],
            self.cfgs[i]["retargeting"]["target_link_names"],
            self.cfgs[i]["retargeting"]["target_link_human_indices"],
        ) for i in range(len(self.robot_names))]
        for opt, wrapper in zip(self.Optimizers, self.robot_wrappers):
            opt.set_joint_limit(wrapper.joint_limits[opt.idx_pin2target])
        self.Finger_Optimizers = [FingerOptimizer(
            hand_name=robot_name.name,
            w_dis=500,
            w_pen=1e9,
            w_joints=1e-2,
        ) for robot_name in self.robot_names]



    def save_meshes_glb(self, robot_meshes, object_mesh, mano_mesh, batch_idx, frame_idx):
        save_dir = "output/mesh_glb"
        os.makedirs(save_dir, exist_ok=True)

        robot_and_object_scene = trimesh.Scene()
        for i, mesh_name in enumerate(robot_meshes):
            mesh = robot_meshes[mesh_name]
            vertices = mesh["vertices"].cpu().numpy()
            faces = mesh["faces"].cpu().numpy()
            vertices = vertices.astype(np.float32)
            faces = faces.astype(np.int32)
            mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh_trimesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile([0.3, 0.3, 0.3, 1.0], (len(vertices), 1)))
            robot_and_object_scene.add_geometry(mesh_trimesh, geom_name=f'robot_part_{i}')
        if hasattr(object_mesh, 'vertices'):
            object_mesh_trimesh = trimesh.Trimesh(object_mesh.vertices, object_mesh.faces)
            vertex_colors = np.tile([1.0, 0.5, 0.0, 1.0] , (len(object_mesh.vertices), 1))
            object_mesh_trimesh.visual = trimesh.visual.ColorVisuals(
                object_mesh_trimesh,
                vertex_colors=vertex_colors
            )
            robot_and_object_scene.add_geometry(object_mesh_trimesh, geom_name='object')
        robot_and_object_scene_path = os.path.join(save_dir, f"robot_and_object_batch{batch_idx}_frame{frame_idx}.glb")
        robot_and_object_scene.export(robot_and_object_scene_path)


        mano_and_object_scene = trimesh.Scene()
        if hasattr(mano_mesh, 'vertices'):
            mano_mesh_trimesh = trimesh.Trimesh(mano_mesh.vertices, mano_mesh.faces)
            vertex_colors = np.tile([0.96, 0.75, 0.65, 1.0], (len(mano_mesh.vertices), 1))  
            mano_mesh_trimesh.visual = trimesh.visual.ColorVisuals(
                mano_mesh_trimesh,
                vertex_colors=vertex_colors
            )
            mano_and_object_scene.add_geometry(mano_mesh_trimesh, geom_name='mano')

        if hasattr(object_mesh, 'vertices'):
            object_mesh_trimesh = trimesh.Trimesh(object_mesh.vertices, object_mesh.faces)
            vertex_colors = np.tile([1.0, 0.5, 0.0, 1.0] , (len(object_mesh.vertices), 1))
            object_mesh_trimesh.visual = trimesh.visual.ColorVisuals(
                object_mesh_trimesh,
                vertex_colors=vertex_colors
            )
            mano_and_object_scene.add_geometry(object_mesh_trimesh, geom_name='object')
        mano_and_object_scene_path = os.path.join(save_dir, f"mano_and_object_batch{batch_idx}_frame{frame_idx}.glb")
        mano_and_object_scene.export(mano_and_object_scene_path)

        print(f"Saved GLB meshes to {save_dir}")

    def retarget(self):
        for idx, data in enumerate(self.dataset):
            mano_joint = data["mano_joint"]
            wrist_quats = data["wrist_quat"]
            dir_index = data["dir_index"]
            object_pcs = data["object_pcs"]
            object_meshes = data["object_meshes"]
            object_poses = data["object_pose"]
            object_names = data["object_name"]
            for Finger_opt,robot_name, optimizer,robot_wrapper,hand_model in zip(self.Finger_Optimizers,
                                                                              self.robot_names,
                                                                              self.Optimizers,
                                                                              self.robot_wrappers,
                                                                              self.hand_models):
                poses_data = []
                last_opt_qpos=None
                scale=SCALE[robot_name.name]
                indices = optimizer.target_link_human_indices
                for i, (joint, quat) in enumerate(zip(mano_joint, wrist_quats)):
                    print(f"\033[31m {idx},{i}\033[0m")
                    #get object information
                    object_pc = object_pcs[i].to(self.device)
                    object_mesh = object_meshes[i]
                    object_pc = object_pc * 1000/scale
                    object_mesh.vertices = object_mesh.vertices * 1000/scale
                    object_name = object_names[0]
                    print()
                    
                    #get mano information
                    joint=joint.cpu().numpy()
                    quat=quat.cpu().numpy()
                    joint = joint * 1000/scale

                    mano_mesh = data["mano_mesh"][i]
                    mano_mesh.vertices = mano_mesh.vertices * 1000/scale
                    object_pose = object_poses[i]
                    self.last_qpos = robot_wrapper.joint_limits.mean(1)[
                        optimizer.idx_pin2target
                    ].astype(np.float32)

                    wrist_quat = rotations.quaternion_from_compact_axis_angle(quat)
                    self.warm_start(
                        joint[0, :],
                        wrist_quat,
                        hand_type=self.hand_type,
                        robot=robot_wrapper,
                        optimizer=optimizer
                    )
                    ref_value = joint[indices, :]
                    fixed_qpos = np.array([])
                    target_q=torch.tensor(self.last_qpos)
                    target_q = optimizer.retarget(ref_value, fixed_qpos, self.last_qpos)
                    self.last_qpos = (
                        target_q.cpu().numpy()
                        if isinstance(target_q, torch.Tensor)
                        else target_q
                    )
                    if not isinstance(target_q, torch.Tensor):
                        target_q = torch.tensor(target_q, dtype=torch.float32)
                    
                    robot_pc_target = hand_model.get_transformed_links_gene_pc(
                        target_q
                    )
                    robot_pc_target = robot_pc_target.to(self.device)
                    link_names = hand_model.link_names

                    new_link_names = []
                    for link_name in link_names:
                        if self.robot_names[0] == RobotName.shadow:
                            if not re.search(r"tip", link_name) and not re.search(r"dummy", link_name) and not  re.search(r"world", link_name) and  not  re.search(r"thhub", link_name) :  # re.search 不需要从头匹配
                                new_link_names.append(link_name)
                        else:
                            if not re.search(r"dummy", link_name) and not  re.search(r"world", link_name):
                                new_link_names.append(link_name)

                    link_names = new_link_names
                    all_link_indices =robot_pc_target[:, 3].long()
                    all_link_names = [link_names[idx] for idx in all_link_indices.tolist()]
                    self.contact_mapping.set_up(robot_pc=robot_pc_target,
                                                    object_pc=object_pc,
                                                    mask_names=self.get_mask_names(robot_name),
                                                    link_names=link_names,
                                                   )
                                                      
                    
                    indices_dict,mask_dict,robot_indices_dict=self.contact_mapping.get_conatct_map_single_frame()
                    print(target_q)
                    print(object_mesh)
                    print(robot_indices_dict)
                    print(indices_dict)
                    Finger_opt.set_up(
                        hand_model=hand_model,
                        object_surface=object_mesh,
                        contact_points_per_finger=robot_indices_dict,
                        finger_joints_indices=self.finger_joints_indices[robot_name.name],
                        initial_hand_pose=target_q,
                        object_contact_point_indices=indices_dict,
                        object_pcs=object_pc,
                    )

                    optimized_hand_pose,finger_scores = Finger_opt.optimize_all_fingers(last_opt_qpos)

                    mean_finger_score=torch.mean(torch.tensor(finger_scores,dtype=torch.float32))
                    if i % 1 == 0:
                        # pass
                        # self.save_meshes_glb(
                        #         robot_meshes=hand_model.get_transformed_links_gene_pc(optimized_hand_pose, return_meshes=True),
                        #         mano_mesh=mano_mesh,  # 请确保定义了mano_mesh或相应的人手Mesh
                        #         object_mesh=object_mesh,
                        #         batch_idx=idx,
                        #         frame_idx=i
                        #     )
                        visualize_and_save_transformed_links_and_object_pc(
                            robot_pc=hand_model.get_transformed_links_gene_pc(
                                optimized_hand_pose
                            ),
                            robot_pc_origin=robot_pc_target,
                            robot_mesh=hand_model.get_transformed_links_gene_pc(
                                optimized_hand_pose, return_meshes=True
                            ),
                            object_mesh=object_mesh,
                            object_pc=object_pc,
                            mano_mesh=mano_mesh,
                            contact_point_indices=robot_indices_dict["all"][0],
                            object_contact_point_indices=indices_dict["all"],
                        )
                    last_opt_qpos=optimized_hand_pose.clone()
                    poses_data.append(
                        {
                            "optimized_hand_pose": optimized_hand_pose,
                            "object_pose": object_pose,
                            "object_name": object_name,
                            "mean_finger_score":mean_finger_score,
                            "finger_scores":finger_scores
                        }
                    )
                self.save_pose_data(poses_data, dir_index)
    def get_mask_names(self,robot_name=RobotName.allegro):
        if robot_name==RobotName.shadow:
            mask_names=["ffdistal","mfdistal","rfdistal","thdistal","lfdistal"]
        elif robot_name==RobotName.allegro:
            mask_names=["link_3.0_tip","link_7.0_tip","link_11.0_tip","link_15.0_tip","None"]
        return mask_names
    def get_contact_map(self,robot_pc_target,object_pc,all_link_names,dis=0.0005,robot_name=RobotName.allegro):
        mask_names=self.get_mask_names(robot_name)
        # mask_all = [bool(re.search(mask_names[0], name, re.IGNORECASE)) for name in all_link_names]
        target_points_all = robot_pc_target[:, :3].to(self.device)
        distance_all, object_contact_point_indices_all = sided_distance(target_points_all.unsqueeze(0),object_pc.to(self.device).unsqueeze(0))
        ff_mask = torch.tensor([bool(re.search(mask_names[0], name, re.IGNORECASE)) for name in all_link_names], dtype=torch.bool, device=self.device)
        mf_mask = torch.tensor([bool(re.search(mask_names[1], name, re.IGNORECASE)) for name in all_link_names], dtype=torch.bool, device=self.device)
        rf_mask = torch.tensor([bool(re.search(mask_names[2], name, re.IGNORECASE)) for name in all_link_names], dtype=torch.bool, device=self.device)
        lf_mask = torch.tensor([bool(re.search(mask_names[4], name, re.IGNORECASE)) for name in all_link_names], dtype=torch.bool, device=self.device)
        th_mask = torch.tensor([bool(re.search(mask_names[3], name, re.IGNORECASE)) for name in all_link_names], dtype=torch.bool, device=self.device)
        target_points_ff = robot_pc_target[ff_mask, :3].to(self.device)
        target_points_mf = robot_pc_target[mf_mask, :3].to(self.device)
        target_points_rf = robot_pc_target[rf_mask, :3].to(self.device)
        target_points_lf = robot_pc_target[lf_mask, :3].to(self.device)
        target_points_th = robot_pc_target[th_mask, :3].to(self.device)

        distance_ff, object_contact_point_indices_ff = sided_distance(target_points_ff.unsqueeze(0),object_pc.to(self.device).unsqueeze(0))
        distance_mf, object_contact_point_indices_mf = sided_distance(target_points_mf.unsqueeze(0),object_pc.to(self.device).unsqueeze(0))
        distance_rf, object_contact_point_indices_rf = sided_distance(target_points_rf.unsqueeze(0),object_pc.to(self.device).unsqueeze(0))
        distance_lf, object_contact_point_indices_lf = sided_distance(target_points_lf.unsqueeze(0),object_pc.to(self.device).unsqueeze(0))
        distance_th, object_contact_point_indices_th = sided_distance(target_points_th.unsqueeze(0),object_pc.to(self.device).unsqueeze(0))
        distance_mask_all=torch.tensor( distance_all < dis).squeeze(0)  # 直接对整个张量进行逐元素比较)
        print("distance_mask_all",torch.tensor(distance_mask_all).shape)
        distance_mask_ff=[distance<dis for distance in distance_ff]
        distance_mask_mf=[distance<dis for distance in distance_mf]
        distance_mask_rf=[distance<dis for distance in distance_rf]
        distance_mask_lf=[distance<dis for distance in distance_lf]
        distance_mask_th=[distance<dis for distance in distance_th]
        object_contact_point_indices_all=object_contact_point_indices_all.squeeze(0)
        object_contact_point_indices_ff=object_contact_point_indices_ff.squeeze(0)
        object_contact_point_indices_mf=object_contact_point_indices_mf.squeeze(0)
        object_contact_point_indices_rf=object_contact_point_indices_rf.squeeze(0)
        object_contact_point_indices_lf=object_contact_point_indices_lf.squeeze(0)
        object_contact_point_indices_th=object_contact_point_indices_th.squeeze(0)
        object_contact_point_indices_all_filtered = object_contact_point_indices_all[distance_mask_all]
        object_contact_point_indices_ff_filtered = object_contact_point_indices_ff[distance_mask_ff]
        object_contact_point_indices_mf_filtered = object_contact_point_indices_mf[distance_mask_mf]
        object_contact_point_indices_rf_filtered = object_contact_point_indices_rf[distance_mask_rf]
        object_contact_point_indices_lf_filtered = object_contact_point_indices_lf[distance_mask_lf]
        object_contact_point_indices_th_filtered = object_contact_point_indices_th[distance_mask_th]
        object_pc=object_pc.to(self.device)
        print("object_contact_point_indices_all_filtered",torch.tensor(object_contact_point_indices_all_filtered).shape)
        distance_all_reverse, robot_contact_point_indices_all = sided_distance(object_pc[object_contact_point_indices_all_filtered].to(self.device).unsqueeze(0),robot_pc_target[:,:3].unsqueeze(0))
        robot_contact_point_mask = torch.zeros(robot_pc_target.shape[0], dtype=torch.bool, device=self.device)
        robot_contact_point_mask[robot_contact_point_indices_all] = True
        ff_mask_reverse = ff_mask & robot_contact_point_mask
        print(th_mask,robot_contact_point_mask)
        print("ff_mask 中 True 的数量:", th_mask.sum())
        print("robot_contact_point_mask 中 True 的数量:", robot_contact_point_mask.sum(),robot_contact_point_indices_all.shape,robot_pc_target.shape)
        mf_mask_reverse = mf_mask & robot_contact_point_mask
        rf_mask_reverse = rf_mask & robot_contact_point_mask
        lf_mask_reverse = lf_mask & robot_contact_point_mask
        th_mask_reverse = th_mask & robot_contact_point_mask
        robot_contact_point_indices_ff = torch.where(ff_mask_reverse)[0] 
        robot_contact_point_indices_mf = torch.where(mf_mask_reverse)[0]
        robot_contact_point_indices_rf = torch.where(rf_mask_reverse)[0]
        robot_contact_point_indices_lf = torch.where(lf_mask_reverse)[0]
        robot_contact_point_indices_th = torch.where(th_mask_reverse)[0]
        
        indices_dict={
            "all":object_contact_point_indices_all_filtered,
            "ff":object_contact_point_indices_ff_filtered,
            "mf":object_contact_point_indices_mf_filtered,
            "rf":object_contact_point_indices_rf_filtered,
            # "lf":object_contact_point_indices_lf_filtered,
            "th":object_contact_point_indices_th_filtered,
        }
        map_dict={
            "ff":ff_mask,
            "mf":mf_mask,
            "rf":rf_mask,
            # "lf":lf_mask,
            "th":th_mask,
        }
        robot_indices_dict={
            "all":robot_contact_point_indices_all,
            "ff":robot_contact_point_indices_ff,
            "mf":robot_contact_point_indices_mf,
            "rf":robot_contact_point_indices_rf,
            # "lf":robot_contact_point_indices_lf,
            "th":robot_contact_point_indices_th,
        }
        if mask_names[4]!="None":
            indices_dict["lf"]=object_contact_point_indices_lf_filtered
            map_dict["lf"]=lf_mask
            robot_indices_dict["lf"]=robot_contact_point_indices_lf
        return indices_dict, map_dict,robot_indices_dict
   
    def get_finger_to_links(self,robot_name: RobotName = RobotName.shadow):
        if robot_name == RobotName.shadow:
            print("shadow")
            return {
                "th": ["thproximal", "thhub", "thmiddle", "thdistal", "thtip"],
                "ff": ["ffproximal", "ffmiddle", "ffdistal", "fftip"],
                "mf": ["mfproximal", "mfmiddle", "mfdistal", "mftip"],
                "rf": ["rfproximal", "rfmiddle", "rfdtistal", "rftip"],
                "lf": [
                    "lfmetacarpal",
                    "lfproximal",
                    "lfmiddle",
                    "lfdistal",
                    "lftip",
                ],
            }
        elif robot_name == RobotName.allegro:
            print("allegro")
            return {
                "th": ["link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"],
                "ff": ["link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"],
                "mf": ["link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"],
                "rf": ["link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"],
            }


    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        robot: RobotWrapper = None,
        optimizer=None,
    ):
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = rotations.matrix_from_quaternion(wrist_quat)
        target_wrist_pose[:3, 3] = wrist_pos
        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]
        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0
        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist
        euler = rotations.euler_from_matrix(
            target_root_pose[:3, :3], 0, 1, 2, extrinsic=False
        )
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])
        for num, joint_name in enumerate(optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]
        self.is_warm_started = True


    def save_pose_data(self, poses_data, batch_idx):
        save_dir = "output/opt_poses_json"
        os.makedirs(save_dir, exist_ok=True)
        data = []
 
        data=tensor_to_list(poses_data)
        json_path = os.path.join(save_dir, f"poses_batch{batch_idx}.json")
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

import argparse
import sys
from enum import Enum

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运动重定向配置工具')
    parser.add_argument('-r', '--robot_names', 
                        nargs='+', 
                        required=True,
                        choices=[name.lower() for name in RobotName.__members__],
                        help='机器人名称列表，可选: {}'.format([name.lower() for name in RobotName.__members__]))
    parser.add_argument('-ht', '--hand_type',
                        required=True,
                        choices=['right', 'left'],
                        help='手部类型，可选：right/left')
    return parser.parse_args()


def main():
    args = parse_args()
    robot_names = [RobotName[name.lower()] for name in args.robot_names]
    hand_type = HandType[args.hand_type.lower()]
    retarget_wrapper = RetargetWrapper(robot_names, hand_type)
    retarget_wrapper.retarget()
    


if __name__ == "__main__":
    main()