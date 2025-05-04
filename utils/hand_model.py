import os
import sys
import json
import math
import random
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.func_utils import farthest_point_sampling
from utils.mesh_utils import load_link_geometries
from utils.rotation import *
from utils.rot6d import *

import torch
from torch.nn import Module


from kaolin.ops.mesh import index_vertices_by_faces, check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
def compose_transformation_matrix(global_translation, global_rotation):
    """
    Compose a 4x4 transformation matrix using rotation and translation.
    
    Args:
        global_translation: Tensor of shape (N, 3)
        global_rotation: Tensor of shape (N, 3, 3)
        
    Returns:
        transformation_matrix: Tensor of shape (N, 4, 4)
    """
    N = global_translation.shape[0]
    
    # Initialize 4x4 identity matrices
    transformation_matrix = torch.eye(4).repeat(N, 1, 1)  # Shape: (N, 4, 4)
    
    # Assign rotation (top-left 3x3)
    transformation_matrix[:, 0:3, 0:3] = global_rotation  # Insert rotation
    
    # Assign translation (top-right 3x1)
    transformation_matrix[:, 0:3, 3] = global_translation  # Insert translation
    
    return transformation_matrix


def get_all_link_names(chain):
    """
    Recursively traverse the Chain object to collect all link names.
    :param chain: Root of the Chain object.
    :return: List of link names.
    """
    link_names = []

    def traverse(node):
        # Append the name of the current node's link if it exists
        if hasattr(node, 'link') and node.link and hasattr(node.link, 'name'):
            link_names.append(node.link.name)
        
        # Recur for all child nodes
        if hasattr(node, 'children') and isinstance(node.children, list):
            for child in node.children:
                traverse(child)

    # Start traversal from the root of the chain
    traverse(chain._root)
    return link_names

class HandModel:
    def __init__(
        self,
        robot_name,
        hand_type,
        urdf_path,
        meshes_path,
        links_pc_path,
        device,
        link_num_points=512
    ):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.meshes_path = meshes_path
        self.device = device
        self.pk_chain = pk.build_chain_from_urdf(open(f"/home/lightcone/workspace/Dexflow/data/urdf/dex_retargeting/hands/{robot_name}_hand/{robot_name}_hand_{hand_type}_glb_w_dummy.urdf").read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.dof_name=self.pk_chain.get_joint_parameter_names()
        if os.path.exists(links_pc_path):  
            links_pc_data = torch.load(links_pc_path, map_location=device)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = None
            self.links_pc_original = None

        self.meshes = load_link_geometries(robot_name, f"/home/lightcone/workspace/Dexflow/data/urdf/dex_retargeting/hands/{robot_name}_hand/{robot_name}_hand_{hand_type}_glb_w_dummy.urdf", get_all_link_names(self.pk_chain))
        self.vertices = {}
        removed_links = json.load(open(os.path.join(ROOT_DIR, 'data_utils/removed_links.json')))[robot_name]
        self.link_names=get_all_link_names(self.pk_chain)
        self.link_names = [link_name for link_name in self.link_names if link_name not in removed_links]
        for link_name, link_mesh in self.meshes.items():
            if link_name in removed_links: 
                continue
            v = link_mesh.sample(link_num_points)
            self.vertices[link_name] = v
        self.frame_status = None

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]
    
    def manage_link_names(self, robot_name):
        link_names = self.link_names
        new_link_names = []
        for link_name in link_names:
            if robot_name == "shadow":
                if not re.search(r"tip", link_name) and not re.search(r"dummy", link_name) and not  re.search(r"world", link_name) and  not  re.search(r"thhub", link_name) :  # re.search 不需要从头匹配
                    new_link_names.append(link_name)
            elif robot_name == "allegro":
                if not re.search(r"dummy", link_name) and not  re.search(r"world", link_name):
                    new_link_names.append(link_name)
        return new_link_names

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q = q.to(torch.float32)

        self.frame_status = self.pk_chain.forward_kinematics(q.to(self.device))

    def get_transformed_links_pc(self, q=None, links_pc=None):
        """
        Use robot link pc & q value to get point cloud.

        :param q: (6 + DOF,), joint values (euler representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (N, 4), with link index
        """
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc
        if q.ndim == 1:
            q = q.unsqueeze(0)

        # self.global_translation = q[:, 0:3].squeeze(0)
        # self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9]).squeeze(0)
        # root_pose_matrix=compose_transformation_matrix(self.global_translation, self.global_rotation).squeeze(0)
        all_pc_se3 = []
        link_names = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            link_names.append(link_name)
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]

            se3_local = self.frame_status[link_name].get_matrix()[0].to(q.device)  
           # 确保 link_pc 和 ones 张量在同一个设备上
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), torch.ones(n_link, 1, device=q.device)], dim=1)

            link_pc_local = (link_pc_homogeneous @ se3_local.T)[:, :3] 

            # link_pc_global = link_pc_local @ self.global_rotation.T + self.global_translation

            # center = link_pc_global.mean(dim=0).tolist()

            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)

            link_pc_global_indexed = torch.cat([link_pc_local, index_tensor], dim=1)
            all_pc_se3.append(link_pc_global_indexed)
        
        self.link_names=link_names
        all_pc_se3 = torch.cat(all_pc_se3, dim=0)
        self.all_pc_se3=all_pc_se3
        return all_pc_se3

    def get_transformed_links_pc1(self, q=None, links_pc=None, visualize=True,
                                internal_sample=True, voxel_res=50, 
                                surface_ratio=0.5,  # 保留ratio参数用于兼容
                                num_surface_points=None,  # 新增表面点数量参数
                                num_internal_points=None, # 新增内部点数量参数
                                jitter_scale=0.5, multi_sample=2, offset_distance=0.003,
                                monte_carlo_attempts=3, total_points=5000):  # 原num_points改为total_points
    
        global np  # 确保全局访问numpy

        def sample_pymesh(mesh, num_points):
            """精确表面采样替代方案"""
            # 自动细分网格提高采样质量
            subdivided = mesh.subdivide_to_size(max_edge=mesh.scale/20)
            
            # 分阶段采样确保数量
            points, _ = trimesh.sample.sample_surface(
                subdivided,
                count=int(num_points * 1.5)  # 过采样150%
            )
            
            # 均匀化处理
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_points)
            kmeans.fit(points)
            return kmeans.cluster_centers_
        # 参数初始化
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc
        
        all_pc_se3 = []
        link_names = []
        total=0
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            # === 初始化变量 ===
            if isinstance(link_pc, torch.Tensor):
                original_pc = link_pc.cpu().numpy().copy()  # 原始表面点
            else:
                original_pc = link_pc.copy()
            combined_pc = original_pc.copy()
            is_internal = np.zeros(len(original_pc), dtype=np.float32)
            
            # === 增强型内部点采样 ===
            if internal_sample and link_name in self.meshes:
                try:
                    mesh = self.meshes[link_name]
                    n_total = total_points
            
                    # 0. 预处理：边界膨胀（解决薄壁问题）
                    try:
                        offset_mesh = mesh.offset(offset_distance)
                        process_mesh = offset_mesh if offset_distance > 0 else mesh
                    except:
                        process_mesh = mesh  # 回退到原网格

                    # 1. 自适应体素分辨率（可选替换固定voxel_res）
                    # voxel_res = self.auto_voxel_res(mesh, n_total)

                    # 2. 生成候选点（含多重采样机制）
                    voxel_grid = process_mesh.voxelized(voxel_res/1000)  # mm转m
                    candidate_points = np.empty((0,3))
                    
                    if not voxel_grid.is_empty:
                        voxel_points = voxel_grid.points.copy()
                        for _ in range(max(1, multi_sample)):  # 多重采样循环
                            # 带幅度的随机抖动
                            jitter = (np.random.rand(*voxel_points.shape) - 0.5) * \
                                    (voxel_grid.scale * jitter_scale)
                            candidate_points = np.concatenate([
                                candidate_points,
                                voxel_points + jitter
                            ])
                        
                        # 3. 精确筛选内部点
                        inside_mask = process_mesh.contains(candidate_points)
                        valid_points = candidate_points[inside_mask]
                        
                        # 4. 蒙特卡洛补充采样（当体素化不足时）
                        if len(valid_points) < n_total * (1 - surface_ratio) * 0.5:
                            mc_points = self.monte_carlo_sampling(
                                process_mesh, 
                                n_points=int(n_total * 2),
                                max_attempts=monte_carlo_attempts
                            )
                            valid_points = np.concatenate([valid_points, mc_points])
                        
                        # 5. 动态配额调整
                        n_surface = max(1, int(n_total * surface_ratio))
                        n_surface = 2500//len(self.link_names)
                        n_internal_target = n_total//len(self.link_names) - n_surface
                        
                        if len(valid_points) >= n_internal_target:
                            selected_points = valid_points[:n_internal_target]
                        elif len(valid_points) > 0:
                            selected_points = valid_points
                            n_surface = n_total - len(selected_points)
                        else:
                            selected_points = np.empty((0,3))
                            n_surface = n_total

                        surface_points = sample_pymesh(mesh, n_surface)

                        # 6. 合并点云
                        combined_pc = np.concatenate([
                            surface_points,  # 表面点部分
                            selected_points           # 验证后的内部点
                        ])
                        
                        # 7. 创建内部标识
                        is_internal = np.concatenate([
                            np.zeros(n_surface),
                            np.ones(len(selected_points))
                        ]).astype(np.float32)

                        # 8. 采样质量检查
                        actual_ratio = n_surface / n_total
                        if abs(actual_ratio - surface_ratio) > 0.2:
                            print(f"[Adjust] {link_name}比例调整: 目标{surface_ratio:.0%} → 实际{actual_ratio:.0%}")
                    
                    # 9. 后处理：均匀性优化
                    combined_pc, is_internal = self.poisson_post_process(
                        combined_pc, 
                        is_internal,
                        surface_ratio=actual_ratio
                    )

                except Exception as e:
                    print(f"[Error] {link_name}采样失败: {str(e)}")
                    combined_pc = original_pc
                    is_internal = np.zeros(len(original_pc))

            # === 坐标变换 ===
            pc_tensor = torch.tensor(combined_pc, dtype=torch.float32, device=q.device)
            n_points = pc_tensor.shape[0]
            
            # 获取变换矩阵
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            
            # 齐次坐标变换
            homogeneous = torch.ones(n_points, 1, device=q.device)
            pc_homo = torch.cat([pc_tensor, homogeneous], dim=1)
            transformed_pc = (pc_homo)[:, :3]

            # === 数据封装 ===
            index_col = torch.full([n_points, 1], float(link_index), device=q.device)
            internal_col = torch.tensor(is_internal[:index_col.shape[0]], device=q.device).reshape(-1,1)
            # link_pc_se3 = torch.cat([transformed_pc, index_col, internal_col], dim=1)
            link_pc_se3=transformed_pc
            all_pc_se3.append(link_pc_se3)
            link_names.append(link_name)

        all_pc_se3 = torch.cat(all_pc_se3, dim=0)

        # === 可视化 ===
        if visualize:
            self.visualize_pointcloud(all_pc_se3)
        return all_pc_se3  # shape: [N,5]

    # 新增辅助函数 --------------------------------------------------
    def auto_voxel_res(self, mesh, target_points):
        """自适应体素分辨率计算"""
        volume = mesh.volume
        if volume < 1e-6: return 50  # 防止零除
        avg_density = target_points / volume
        res = int((1 / avg_density)**(1/3)*1000)
        return np.clip(res, 20, 100)  # 限制在20-100mm之间

    def monte_carlo_sampling(self, mesh, n_points=1000, max_attempts=5):
        """蒙特卡洛拒绝采样"""
        selected = np.empty((0,3))
        bounds = mesh.bounds
        for _ in range(max_attempts):
            rand_points = np.random.rand(n_points,3) * (bounds[1]-bounds[0]) + bounds[0]
            inside = mesh.contains(rand_points)
            selected = np.concatenate([selected, rand_points[inside]])
            if len(selected) >= n_points: break
        return selected[:n_points]

    def poisson_post_process(self, points, is_internal, surface_ratio, min_spacing=0.01):
        """泊松盘采样后处理"""
        try:
            from sklearn.neighbors import NearestNeighbors
            # 分离表面点和内部点
            surface_mask = is_internal < 0.5
            surface_points = points[surface_mask]
            internal_points = points[~surface_mask]
            
            # 仅对内部点进行均匀化
            if len(internal_points) > 10:
                nbrs = NearestNeighbors(radius=min_spacing).fit(internal_points)
                clusters = nbrs.radius_neighbors_graph()
                # 实现聚类筛选（此处简化处理）
                filtered_points = internal_points[::2]  # 每间隔一个点采样
                
                # 重组点云
                new_points = np.concatenate([surface_points, filtered_points])
                new_labels = np.concatenate([
                    np.zeros(len(surface_points)),
                    np.ones(len(filtered_points))
                ])
                return new_points, new_labels
            return points, is_internal
        except:
            return points, is_internal

    def visualize_pointcloud(self, pc_tensor):
        """可视化增强"""
        try:
            import open3d as o3d
            pc_np = pc_tensor.cpu().numpy()
            
            # 高级颜色映射
            colors = np.zeros((len(pc_np),3))
            # internal_mask = pc_np[:,4] > 0.5
            # colors[internal_mask] = [1, 0.2, 0.2]  # 深红色：内部点
            # colors[~internal_mask] = [0.8, 0.8, 0.8]  # 浅灰色：表面点
            colors[:]= [1, 0.2, 0.2]
            # 创建带法向量的点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_np[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.estimate_normals()
            
            # 添加语义标注
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (0, 1, 0)
            
            # 可视化界面
            o3d.visualization.draw_geometries([pcd, coord, bbox],
                                            window_name="Sampling Preview",
                                            width=1024, height=768)
        except ImportError:
            print("可视化需要Open3D，请执行: pip install open3d")

    def compute_single_link_local_jacobian(self, link_name: str, q: torch.Tensor = None):
        # 输入参数检查
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        elif q.shape != (self.dof,):
            raise ValueError(f"Joint positions q must be of shape ({self.dof},), got {q.shape}")
        
        # 更新运动学链状态
        self.update_status(q)
        
        # 查找目标 link/frame
        frame = self.pk_chain.find_frame(link_name)
        if frame is None:
            available_frames = self.pk_chain.get_frame_names()
            raise ValueError(f"Link '{link_name}' not found. Available frames: {available_frames}")
        
        frame_status = self.frame_status[link_name]  # 假设 frame_status 按索引存储
        
        # 计算雅可比矩阵
        jacobian =self.compute_jacobian_via_autograd(
            frame_status,
            q
        )
        
        return jacobian

    def compute_jacobian_via_autograd(self, transform, q):
        matrix = transform.get_matrix()  
        translation = matrix[..., :3, 3]
        pose_tensor = translation.flatten()  # 或者 translation.view(-1)
    
        jacobian = []
        for i in range(pose_tensor.numel()):
            grad_outputs = torch.zeros_like(pose_tensor)
            grad_outputs[i] = 1.0
            # 计算梯度，确保 q 需要梯度
            grad_i, = torch.autograd.grad(
                outputs=pose_tensor,
                inputs=q,
                grad_outputs=grad_outputs,
                retain_graph=True,
                allow_unused=False
            )
            jacobian.append(grad_i)
        return torch.stack(jacobian).view(pose_tensor.numel(), -1)
    def get_link_pose(self, link_name, q=None):
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        return self.frame_status[link_name].get_matrix()[0]
    def get_transformed_links_gene_pc(self, q=None, links_pc=None,return_meshes=False,return_distal_centroid=False):
        """
        Use robot link pc & q value to get point cloud.

        :param q: (6 + DOF,), joint values (euler representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (N, 4), with link index
        """
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc

        all_pc_se3 = []
        transformed_meshes = {}  
        distal_points = {}
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]

            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)

            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)
            # 对 meshes 的变换
            if link_name in self.meshes:

                mesh = self.meshes[link_name]
                vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=q.device)
                faces = torch.tensor(mesh.faces, dtype=torch.long, device=q.device)
                homogeneous_vertices = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=q.device)], dim=1)
                transformed_vertices = (homogeneous_vertices @ se3.T)[:, :3]
                transformed_meshes[link_name] = {"vertices": transformed_vertices, "faces": faces}
            if re.match(r".*distal$", link_name):
                distal_points.setdefault(link_name, []).append(link_pc_se3)

        # ==== 结果整合 ====
        all_pc_se3 = torch.cat(all_pc_se3, dim=0).to(q.device)  # (Total_N,4)
        self.transformed_meshes = transformed_meshes
        self.all_pc_se3 = all_pc_se3

        # ==== 返回处理 ====
        if return_distal_centroid and distal_points:
            distal_centroids = []
            for points in distal_points.values():
                combined = torch.cat(points, dim=0)  # 合并同关节不同批次点云
                distal_centroids.append(combined.mean(dim=0))  # 计算质心
                
            distal_centroids = torch.stack(distal_centroids)  # (K,3)
            if return_meshes:
                return transformed_meshes, distal_centroids.unsqueeze(0)  # (1,K,3)
            return all_pc_se3, distal_centroids.unsqueeze(0)

        if return_meshes:
            return  transformed_meshes
        return all_pc_se3


    def self_penetration(self, current_finger_keypoints, other_fingers_keypoints):
        """
        Calculate self penetration energy between one finger and other fingers.

        Parameters
        ----------
        current_finger_keypoints : torch.Tensor
            Shape: (N, K1, 3) where N = batch size, K1 = keypoints for the current finger
        other_fingers_keypoints : torch.Tensor
            Shape: (N, K2, 3) where N = batch size, K2 = keypoints for other fingers

        Returns
        -------
        E_spen : (N,) torch.Tensor
            Self penetration energy
        """
        # Calculate pairwise distances between current finger and other fingers
        dis = (current_finger_keypoints.unsqueeze(2) - other_fingers_keypoints.unsqueeze(1) + 1e-13).square().sum(3).sqrt()

        # Avoid zero distances (self-comparisons are not relevant here since it's two sets of points)
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)

        # Self-penetration loss (distance less than threshold penalized)
        loss = -torch.where(dis < 0.0018, dis, torch.zeros_like(dis))

        # Sum the losses over all keypoint pairs for each batch
        return loss.sum((1, 2))
    
    def get_pd(self, obj_surface):
        obj_faces = torch.tensor(obj_surface.faces, dtype=torch.long, device="cuda:0")
        obj_vertices = torch.tensor(obj_surface.vertices, dtype=torch.float32, device="cuda:0")
        obj_faces_vertices = obj_vertices[obj_faces].contiguous()
        
        squared_distance, distance_signs, normals, closest_points = self.kaolin_sdf(obj_vertices, obj_faces)
        product = squared_distance * distance_signs
        min_value = torch.min(product)
        
        return min_value
    
    def get_spd(self,robot_mesh):
        robot_faces = torch.tensor(robot_mesh.faces, dtype=torch.long, device="cuda:0")
        robot_vertices = torch.tensor(robot_mesh.vertices, dtype=torch.float32, device="cuda:0")
        squared_distance, distance_signs, normals, closest_points = self.kaolin_sdf(robot_vertices, robot_faces)
        product = squared_distance * distance_signs
        min_value = torch.min(product)
        return min_value

        
        


    def get_contact_point_candidates(self, obj_surface):
        obj_faces = torch.tensor(obj_surface.faces, dtype=torch.long, device="cuda:0")
        obj_vertices = torch.tensor(obj_surface.vertices, dtype=torch.float32, device="cuda:0")
        obj_faces_vertices = obj_vertices[obj_faces].contiguous()
        
        # 确保SDF计算结果在GPU上
        squared_distance, distance_signs, normals, closest_points = self.kaolin_sdf(obj_vertices, obj_faces)
        
        # 条件判断
        condition = torch.logical_and(
            (squared_distance * distance_signs) < 0.00001,
            (squared_distance * distance_signs) > -0.0005
        )
        
        # 获取满足条件的索引（确保结果在GPU）
        indices = torch.nonzero(condition, as_tuple=True)[0]
        
        # 关键修改：确保索引和点云在相同设备
        if self.all_pc_se3.device != indices.device:
            self.all_pc_se3 = self.all_pc_se3.to(indices.device)
        
        # 现在可以安全地进行索引操作
        link_indices = self.all_pc_se3[indices, 3].long()
        sdf_values = squared_distance[indices] * distance_signs[indices]
        
        selected_indices = []
        
        for link_id, link_name in enumerate(self.link_names):
            link_mask = (link_indices == link_id)
            link_points = indices[link_mask]
            link_sdf_values = sdf_values[link_mask]
            
            sorted_indices = torch.argsort(link_sdf_values)
            if re.match(r".*distal$", link_name):
                selected_points = link_points[sorted_indices[:30]]
            else:
                selected_points = link_points[sorted_indices[:10]]
            
            selected_indices.append(selected_points)
        
        selected_indices = torch.cat(selected_indices, dim=0)
        
        return selected_indices

    
    def cal_distance_keypoint(self, keypoint_index, mesh):
        """
        计算物体点云到手部网格的有符号距离。

        Parameters
        ----------
        keypoint_index : int
            点云的关键点索引。
        mesh : trimesh.Trimesh
            用于计算距离的手部网格。
        """
        dis = []
        
        # 提取关键点
        x_keypoint = torch.tensor(self.all_pc_se3[:, :3], dtype=torch.float32)

        # 确保 x_keypoint 在 CUDA 上
        x_keypoint = x_keypoint.cuda()

        # 转换 Trimesh 网格为 CUDA Tensor
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()  # 顶点坐标
        faces = torch.tensor(mesh.faces, dtype=torch.int64).cuda()  # 面索引

        # 根据面索引提取三角形顶点坐标
        face_vertices = vertices[faces]  # 转换为形状 (num_faces, 3, 3)

        # 调用 compute_sdf，计算有符号距离
        dis_local, dis_signs, _, _ = self.kaolin_sdf(x_keypoint, face_vertices)


        
    def add_noise_to_pc(self, pc, noise_level=0.01):
        """
        Add noise to the input point cloud.

        :param pc: (N, 3), point cloud
        :param noise_level: float, noise level
        :return: (N, 3), point cloud with noise
        """
        noise = torch.randn_like(pc) * noise_level
        return pc + noise
        

    def get_sampled_pc(self, q=None, num_points=512):
        """
        :param q: (9 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()

        sampled_pc = self.get_transformed_links_pc1(q, self.vertices)
        return sampled_pc

    def get_canonical_q(self):
        """ For visualization purposes only. """
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q[:6] = 0
        canonical_q*=0
        return canonical_q

    def get_random_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute the robot initial joint value q based on the target grasp.
        Root translation is not considered since the point cloud will be normalized to zero-mean.

        :param q: (6 + DOF,) or (9 + DOF,), joint values (euler/rot6d representation)
        :param max_angle: float, maximum angle of the random rotation
        :return: initial q: (6 + DOF,), euler representation
        """
        if q is None:  # random sample root rotation and joint values
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        else:
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial rotation
            # direction = - q_initial[:3] / torch.norm(q_initial[:3])
            # angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            # axis = torch.randn(3).to(q.device)  # sample rotation axis
            # axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            # axis = axis / torch.norm(axis)
            # random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            # rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            # q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            noise = torch.tensor(
                [random.uniform(-0.15, 0.15) for _ in range(len(lower_joint_limits))],
                dtype=torch.float32
            )
            q_initial[9:] +=(upper_joint_limits - lower_joint_limits)*noise 
            # q_initial[9:] = torch.zeros_like(q_initial[9:], dtype=q.dtype, device=q.device)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial

    def get_trimesh_q(self, q):
        """ Return the hand trimesh object corresponding to the input joint value q. """
        self.update_status(q)

        scene = trimesh.Scene()
        for link_name in self.vertices:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        parts = {}
        for link_name in self.meshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = self.meshes[link_name].copy().apply_transform(mesh_transform_matrix)
            parts[link_name] = part_mesh

        return_dict = {
            'visual': trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        """ Return the hand trimesh object corresponding to the input transform. """
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            scene.add_geometry(self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))

        vertices = []
        faces = []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)

        return trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    def compute_world_coordinates_and_normals_all(self, hand_pose):
        self.update_status(hand_pose)
        transformed_links_pc = self.get_transformed_links_pc(hand_pose)
        contact_points_world = transformed_links_pc[:, :3].clone()  # 去掉 .detach()
        contact_normals = self.calculate_normals(contact_points_world)
        return contact_points_world, contact_normals 

    def compute_world_coordinates_and_normals(self, contact_points, hand_pose,obj_surface):
        indexed_pc=self.get_transformed_links_gene_pc(hand_pose)
        link_index=indexed_pc[:,3]
        link_pcs=indexed_pc[:,:3]
        squared_distance, distance_signs,normals,  closest_points = self.calculate_normals(obj_surface)
        return squared_distance, distance_signs,normals,  closest_points,link_index,link_pcs

    def calculate_normals(self,obj_surface):
        """
        Calculate normals for the given vertices using the robot's mesh data.
        
        :param vertices: torch.Tensor, (N, 3), input vertices for which normals are calculated.
        :return: torch.Tensor, (N, 3), normals of the vertices.
        """
        obj_faces = torch.tensor(obj_surface.faces, dtype=torch.long, device=self.device)
        obj_vertices = torch.tensor(obj_surface.vertices, dtype=torch.float32, device=self.device)
        obj_faces_vertices = obj_vertices[obj_faces].to("cuda:0")

        

        robot_vertices= self.all_pc_se3[:, :3].clone().to("cuda:0")
        squared_distance, distance_signs,normals,  closest_points = self.kaolin_sdf(obj_vertices, obj_faces)
        return squared_distance, distance_signs,normals,  closest_points

    def compute_world_coordinates_and_normals_kaolin(self, contact_points, hand_pose,vertices,faces):
        indexed_pc=self.get_transformed_links_gene_pc(hand_pose)
        link_index=indexed_pc[:,3]
        link_pcs=indexed_pc[:,:3]
        squared_distance, distance_signs,normals,  closest_points = self.kaolin_sdf(vertices,faces)
        return squared_distance, distance_signs,normals,  closest_points,link_index,link_pcs

    def kaolin_sdf(self,vertices,faces,robot_vertices=None):
        if robot_vertices is None:
            robot_vertices= self.all_pc_se3[:, :3].clone().to("cuda:0")
            robot_vertices = robot_vertices.unsqueeze(0).to("cuda:0")  # Reshape to [1, num_vertices, 3] for a single mesh
        else:
            robot_vertices=torch.tensor(robot_vertices, dtype=torch.float32, device="cuda:0")
            robot_vertices=robot_vertices.to("cuda:0")
        faces=faces.to("cuda:0")
        vertices = vertices.unsqueeze(0).to("cuda:0")  # Reshape to [1, num_vertices, 3] for a single mesh
        face_vertices = index_vertices_by_faces(vertices, faces).to("cuda:0")
        squared_distance, index, dist_type= point_to_mesh_distance(robot_vertices, face_vertices)
        inside_mesh = check_sign(vertices, faces, robot_vertices).squeeze(0)
        signs = torch.ones_like(inside_mesh, dtype=torch.int)  # 初始化为 0 的张量
        signs[inside_mesh == True] = -1  # 将 True 的位置设为 -1
        return squared_distance.squeeze(0), signs, index.squeeze(0), dist_type.squeeze(0)

    # def compute_world_coordinates_and_normals_all(self, hand_pose):
    #     self.update_status(hand_pose)
    #     transformed_links_pc = self.get_transformed_links_pc(hand_pose)
    #     contact_points_world = transformed_links_pc[:, :3].clone().detach()
    #     contact_normals = self.calculate_normals(contact_points_world)
    #     return contact_points_world, contact_normals

    # def compute_world_coordinates_and_normals(self, contact_points, hand_pose):
    #     self.update_status(hand_pose)
    #     transformed_links_pc = self.get_transformed_links_pc(hand_pose)
    #     contact_points_world = transformed_links_pc[contact_points, :3].clone().detach()
    #     contact_normals = self.calculate_normals(contact_points_world)
    #     return contact_points_world, contact_normals

    # def calculate_normals(self, vertices):
    #     vertex_normals = torch.zeros(vertices.shape[0], 3, dtype=torch.float32, device=self.device)

    #     for link_name, link_mesh in self.meshes.items():
    #         faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=self.device)
    #         link_vertices = torch.tensor(link_mesh.vertices, dtype=torch.float32, device=self.device)

    #         if torch.any(faces >= len(link_vertices)):
    #             raise ValueError(f"Faces contain indices out of range for vertices in link: {link_name}")

    #         valid_faces_mask = (faces < vertex_normals.shape[0]).all(dim=1)
    #         faces = faces[valid_faces_mask]

    #         if faces.numel() == 0:
    #             continue

    #         v0 = link_vertices[faces[:, 0]]
    #         v1 = link_vertices[faces[:, 1]]
    #         v2 = link_vertices[faces[:, 2]]

    #         normals = torch.cross(v1 - v0, v2 - v0)
    #         normals = torch.nn.functional.normalize(normals, dim=-1)
    #         vertex_normals.index_add_(0, faces[:, 0], normals)
    #         vertex_normals.index_add_(0, faces[:, 1], normals)
    #         vertex_normals.index_add_(0, faces[:, 2], normals)

    #     vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
    #     return vertex_normals



import json
import numpy as np
import torch
import trimesh as tm
from manopth.manolayer import ManoLayer
import plotly.graph_objects as go
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points

class ycb_MANOLayer(Module):
    """Wrapper layer for manopth ManoLayer."""

    def __init__(self, side, betas, mano_root='manopth/mano/models'):
        """Constructor.
        Args:
          side: MANO hand type. 'right' or 'left'.
          betas: A numpy array of shape [10] containing the betas.
        """
        super(ycb_MANOLayer, self).__init__()

        self._side = side
        self._betas = betas
        self._mano_layer = ManoLayer(flat_hand_mean=False,
                                     ncomps=45,
                                     side=self._side,
                                     mano_root=mano_root,
                                     use_pca=True)

        # Ensure betas are converted to a writable tensor and cast to float32
        b = torch.from_numpy(self._betas.copy()).unsqueeze(0).float()  # Add .copy() and .float()
        f = self._mano_layer.th_faces
        self.register_buffer('b', b)
        self.register_buffer('f', f)

        # Precompute root translation
        v = torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(
            0, 1)).permute(2, 0, 1) + self._mano_layer.th_v_template
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer('root_trans', r)

    def forward(self, p, t):
        """Forward function.
        Args:
          p: A tensor of shape [B, 48] containing the pose.
          t: A tensor of shape [B, 3] containing the trans.
        Returns:
          v: A tensor of shape [B, 778, 3] containing the vertices.
          j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)
        # v /= 1000
        # j /= 1000
        return v, j


class ManoModel:
    def __init__(self, mano_root, contact_indices_path, pose_distrib_path, device='cpu'):
        """
        Create a Hand Model for MANO
        
        Parameters
        ----------
        mano_root: str
            base directory of MANO_RIGHT.pkl
        contact_indices_path: str
            path to hand-selected contact candidates
        pose_distrib_path: str
            path to a multivariate gaussian distribution of the `thetas` of MANO
        device: str | torch.Device
            device for torch tensors
        """

        # load MANO

        self.device = device
        # self.manolayer = ManoLayer(mano_root=mano_root, flat_hand_mean=True, use_pca=False).to(device=self.device)
        betas = np.array([
            0.6993994116783142,
            -0.16909725964069366,
            -0.8955091834068298,
            -0.09764610230922699,
            0.07754238694906235,
            0.336286723613739,
            -0.05547792464494705,
            0.5248727798461914,
            -0.38668063282966614,
            -0.00133091164752841
        ])

        # betas = np.array([
        #     -1.1408641338348389,
        #     -0.5010303854942322,
        #     -1.4958895444869995,
        #     -0.6343677043914795,
        #     0.16275525093078613,
        #     0.4472178816795349,
        #     0.15918582677841187,
        #     1.0586552619934082,
        #     0.09903410822153091,
        #     -0.21815064549446106,
        # ])
        self.manolayer = ycb_MANOLayer('right', betas, mano_root).to(device=self.device)

        # load contact points and pose distribution
        
        with open(contact_indices_path, 'r') as f:
            self.contact_indices = json.load(f)
        self.contact_indices = torch.tensor(self.contact_indices, dtype=torch.long, device=self.device)
        self.n_contact_candidates = len(self.contact_indices)

        self.pose_distrib = torch.load(pose_distrib_path, map_location=device)
        
        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.vertices = None
        self.keypoints = None
        self.contact_points = None
    def assign_vertices_to_fingers(self, vertices, joints):
        vertices = vertices.squeeze(0).to(dtype=torch.float32, device="cuda")
        joints = joints.squeeze(0).to(dtype=torch.float32, device=vertices.device)
        def manual_cdist(x1, x2):
            diff = x1.unsqueeze(1) - x2.unsqueeze(0)
            distances = torch.sqrt((diff ** 2).sum(-1))
            return distances
        distances = manual_cdist(vertices, joints)
        closest_joint_indices = torch.argmin(distances, dim=1)
        finger_groups = {
            'th': [1, 2, 3, 4],
            'ff': [5, 6, 7, 8],
            'mf': [9, 10, 11, 12],
            'rf': [13, 14, 15, 16],
            'lf': [17, 18, 19, 20],
        }

        # 分配顶点到手指
        finger_assignments = torch.zeros(vertices.size(0), dtype=torch.long, device=vertices.device)
        link_assignments = torch.zeros(vertices.size(0), dtype=torch.long, device=vertices.device)
        for finger_idx, joint_indices in enumerate(finger_groups.values()):
            mask = torch.isin(closest_joint_indices, torch.tensor(joint_indices, device=vertices.device))
            finger_assignments[mask] = finger_idx
            link_assignments[mask] = joint_indices[0]
        return finger_assignments, link_assignments






    def set_parameters(self, hand_pose, contact_point_indices=None,data_origin="net"):
        """
        Set translation, rotation, thetas, and contact points of grasps
        
        Parameters
        ----------
        hand_pose: (B, 3+3+45) torch.FloatTensor
            translation, rotation in axis angles, and `thetas`
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        if data_origin == "ycb":
            self.vertices, self.joints = self.manolayer(
                self.hand_pose[:, :-3],
                self.hand_pose[:, -3:],
            )
        else:    
            self.vertices, self.joints = self.manolayer(
                th_trans=self.hand_pose[:, :3],
                th_pose_coeffs=self.hand_pose[:, 3:],
            )
        self.vertices /=1000.0
        self.joints /= 1000.0
        self.contact_point_indices = contact_point_indices
        self.contact_points = self.vertices[torch.arange(len(hand_pose)).unsqueeze(1), self.contact_indices[self.contact_point_indices]]
        return self.vertices, self.joints, self.contact_points
    
    def camera_to_world(self, camera_pose):
        camera_mat = camera_pose.to_transformation_matrix()
        self.vertices = self.vertices.cpu().numpy() @ camera_mat[:3, :3].T + camera_mat[:3, 3]
        self.vertices = np.ascontiguousarray(self.vertices)
        self.joints = self.joints.cpu().numpy() @ camera_mat[:3, :3].T + camera_mat[:3, 3]
        self.joints = np.ascontiguousarray(self.joints)
        self.vertices = torch.tensor(self.vertices, dtype=torch.float32, device=self.device)
        self.joints = torch.tensor(self.joints, dtype=torch.float32, device=self.device)
        
    def cal_dis_plane(self, p):
        """
        Calculate signed distances from each MANO vertex to table plane (above+, below-)
        
        Parameters
        ----------
        p: (B, 4) torch.Tensor
            plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
        Returns
        -------
        dis: (B, 778) torch.Tensor
            signed distances from each MANO vertex to table plane
        """
        dis = (p[:, :3].unsqueeze(1) * self.vertices).sum(2) + p[:, 3].unsqueeze(1)
        return dis

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes
        
        Interiors are positive, exteriors are negative
        
        Use the inner product of the ObjectPoint-to-HandNearestNeighbour vector 
        and the vertex normal of the HandNearestNeighbour to approximate the sdf
        
        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Alternative 1: directly using Kaolin results in a time-consuming for-loop along the batch dimension
        # Alternative 2: discarding the inner product with the vertex normal will mess up the optimization severely
        # we reserve the implementation of the second alternative as comments below
        mesh = Meshes(verts=self.vertices, faces=self.manolayer.th_faces.unsqueeze(0).repeat(len(x), 1, 1))
        normals = mesh.verts_normals_packed().view(-1, 778, 3)
        knn_result = knn_points(x, self.vertices, K=1)
        knn_idx = (torch.arange(len(x)).unsqueeze(1), knn_result.idx[:, :, 0])
        dis = -((x - self.vertices[knn_idx]) * normals[knn_idx].detach()).sum(dim=-1)
        # interior = ((x - self.vertices[knn_idx]) * normals[knn_idx]).sum(dim=-1) < 0
        # dis = torch.sqrt(knn_result.dists[:, :, 0] + 1e-8)
        # dis = torch.where(interior, dis, -dis)
        return dis
    
    def kaolin_sdf(self,vertices,faces):
        
  
        robot_vertices= self.vertices.clone().to("cuda:0")
        robot_vertices = robot_vertices

        vertices = vertices.unsqueeze(0).to("cuda:0")  # Reshape to [1, num_vertices, 3] for a single mesh
        faces=faces.to("cuda:0")
        face_vertices = index_vertices_by_faces(vertices, faces).to("cuda:0")

        robot_vertices = robot_vertices.contiguous()
        face_vertices = face_vertices.contiguous()
        squared_distance, index, dist_type= point_to_mesh_distance(robot_vertices, face_vertices)
        inside_mesh = check_sign(vertices, faces, robot_vertices).squeeze(0)
        signs = torch.ones_like(inside_mesh, dtype=torch.int)  # 初始化为 0 的张量
        signs[inside_mesh == True] = -1  # 将 True 的位置设为 -1
        return squared_distance.squeeze(0), signs, index.squeeze(0), dist_type.squeeze(0)


    
    def get_contact_point_candidates(self, obj_surface):
        obj_faces = torch.tensor(obj_surface.faces, dtype=torch.long, device="cuda:0")
        obj_vertices = torch.tensor(obj_surface.vertices, dtype=torch.float32, device="cuda:0")
        obj_faces_vertices = obj_vertices[obj_faces]
        obj_faces_vertices = obj_faces_vertices.contiguous()
        # if isinstance(self.vertices, torch.Tensor):
        #     vertices = self.vertices
        # else:    
        vertices = torch.tensor(self.vertices, dtype=torch.float32, device="cuda:0")
        vertices = vertices.contiguous()
        squared_distance, distance_signs, normals, closest_points = self.kaolin_sdf(obj_vertices,obj_faces)
        indices = torch.nonzero((squared_distance) < 0.000002, as_tuple=True)[0]
        return indices
    
    # def get_contact_point_candidates(self, obj_surface):
    #     obj_faces = torch.tensor(obj_surface.faces, dtype=torch.long, device="cuda:0")
    #     obj_vertices = torch.tensor(obj_surface.vertices, dtype=torch.float32, device="cuda:0")
    #     obj_faces_vertices = obj_vertices[obj_faces]
    #     obj_faces_vertices = obj_faces_vertices.contiguous()

    #     # 确保 vertices 是张量
    #     vertices = self.vertices.to(dtype=torch.float32, device="cuda:0")
    #     vertices = vertices.contiguous()

    #     # 计算 SDF 数据
    #     squared_distance, distance_signs, normals, closest_points = compute_sdf(vertices[0], obj_faces_vertices)

    #     # 筛选满足条件的顶点索引
    #     condition = (squared_distance * distance_signs) < 0.001
    #     indices = torch.nonzero(condition, as_tuple=True)[0]

    #     finger_indices,link_indices = self.assign_vertices_to_fingers(vertices, self.joints)
    #     print(link_indices)
    #     selected_indices = []
    #     for link_id in range(21): 
    #         link_mask = (link_indices == link_id)
    #         link_points = indices[link_mask[indices]]
    #         selected_points = link_points[:5]
    #         selected_indices.append(selected_points)



        # for finger_id in range(5):  # 手指 ID：0-4
        #     # 获取当前手指的点
        #     finger_mask = (finger_indices == finger_id)
        #     finger_points = indices[finger_mask[indices]]

        #     selected_points = finger_points[:15]  
        #     selected_indices.append(selected_points)

        # 合并所有手指的点
        selected_indices = torch.cat(selected_indices, dim=0)

        return selected_indices

    
    def self_penetration(self,keypoints):
        """
        Calculate self penetration energy
        
        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        dis = (keypoints.unsqueeze(1) - keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        loss = -torch.where(dis < 0.028, dis, torch.zeros_like(dis))
        return loss.sum((1,2))

    def get_contact_candidates(self):
        """
        Get all contact candidates
        
        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        return self.vertices[torch.arange(len(self.vertices)).unsqueeze(1), self.contact_indices.unsqueeze(0)]
    
    def get_penetraion_keypoints(self):
        """
        Get MANO keypoints
        
        Returns
        -------
        points: (N, 21, 3) torch.Tensor
            MANO keypoints
        """
        return self.keypoints

    def get_plotly_data(self, i, opacity=0.5, color='lightblue', with_keypoints=False, with_contact_points=False, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_keypoints: bool
            whether to visualize keypoints
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        v = self.vertices[i].detach().cpu().numpy()
        if pose is not None:
            v = v @ pose[:3, :3].T + pose[:3, 3]
        f = self.manolayer.th_faces
        hand_plotly = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], text=list(range(len(v))), color=color, opacity=opacity, hovertemplate='%{text}')]
        if with_keypoints:
            keypoints = self.keypoints[i].detach().cpu().numpy()
            if pose is not None:
                keypoints = keypoints @ pose[:3, :3].T + pose[:3, 3]
            hand_plotly.append(go.Scatter3d(x=keypoints[:, 0], y=keypoints[:, 1], z=keypoints[:, 2], mode='markers', marker=dict(color='red', size=5)))
            for penetration_keypoint in keypoints:
                mesh = tm.primitives.Capsule(radius=0.009, height=0)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                hand_plotly.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            hand_plotly.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2], mode='markers', marker=dict(color='red', size=5)))
        return hand_plotly
    
    def get_trimesh_data(self, i):
        """
        Get visualization data for trimesh
        
        Parameters
        ----------
        i: int
            index of data
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        v = self.vertices[i].detach().cpu().numpy()
        # f = self.manolayer.th_faces.detach().cpu().numpy()
        f = self.manolayer._mano_layer.th_faces.detach().cpu().numpy()
        data = tm.Trimesh(v, f)
        return data
    
    def get_trans_trimesh_data(self, i,pose=None):
        """
        Get visualization data for trimesh
        
        Parameters
        ----------
        i: int
            index of data
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        v = self.vertices[i].detach().cpu().numpy()
        joints = self.joints[i].detach().cpu().numpy()
        # f = self.manolayer.th_faces.detach().cpu().numpy()
        f = self.manolayer._mano_layer.th_faces.detach().cpu().numpy()
        data = tm.Trimesh(v, f)
        return data, joints





def create_hand_model(
    robot_name,
    hand_type,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_points=512
):
    if robot_name!='mano':
        json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta.json')
        urdf_assets_meta = json.load(open(json_path))
        urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
        meshes_path = os.path.join(ROOT_DIR, urdf_assets_meta['meshes_path'][robot_name])
        links_pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/robot/{robot_name}.pt')
        hand_model = HandModel(robot_name,hand_type, urdf_path, meshes_path, links_pc_path, device, num_points)
    elif robot_name=='mano':
        hand_model = ManoModel(
        mano_root='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano', 
        contact_indices_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano/contact_indices.json', 
        pose_distrib_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano/pose_distrib.pt', 
        device=device
        )
    return hand_model
