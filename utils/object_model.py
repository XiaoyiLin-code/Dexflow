"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

import os
import trimesh as tm
import plotly.graph_objects as go
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np
import time
from kaolin.ops.mesh import index_vertices_by_faces, check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance


class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float, device=self.device)




    def initialize(self, object_code_list):
        """
        Initialize Object Model with list of objects
        
        Choose scales, load meshes, sample surface points
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []

        start_time = time.time()  # 记录起始时间

        for object_idx, object_code in enumerate(object_code_list):
            # print(f"Initializing object {object_idx + 1}/{len(object_code_list)}: {object_code}")
            
            # Step 1: 选择缩放比例
            step_time = time.time()
            scale = self.scale_choice[
                torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each,), device=self.device)
            ]
            self.object_scale_tensor.append(scale)
            # print(f"  Step 1 (choose scale) took {time.time() - step_time:.2f} seconds")

            # Step 2: 加载网格
            step_time = time.time()
            mesh_path = os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj")
            mesh = tm.load(mesh_path, force="mesh", process=False)
            self.object_mesh_list.append(mesh)
            # print(f"  Step 2 (load mesh from {mesh_path}) took {time.time() - step_time:.2f} seconds")

            # Step 3: 获取顶点和面
            step_time = time.time()
            object_verts = torch.Tensor(mesh.vertices).to(self.device)
            object_faces = torch.Tensor(mesh.faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            # print(f"  Step 3 (process vertices and faces) took {time.time() - step_time:.2f} seconds")

            # Step 4: 采样点云
            if self.num_samples != 0:
                step_time = time.time()
                vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(mesh.faces, dtype=torch.float, device=self.device)
                pytorch3d_mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                
                # 密集点云采样
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                    pytorch3d_mesh, num_samples=100 * self.num_samples
                )
                # print(f"    Step 4.1 (dense point cloud sampling) took {time.time() - step_time:.2f} seconds")

                step_time = time.time()
                # 最远点采样
                surface_points = pytorch3d.ops.sample_farthest_points(
                    dense_point_cloud, K=self.num_samples
                )[0][0]
                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)
                # print(f"    Step 4.2 (farthest point sampling) took {time.time() - step_time:.2f} seconds")

        # Step 5: 汇总张量
        step_time = time.time()
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        # print(f"Step 5 (stack scale tensor) took {time.time() - step_time:.2f} seconds")

        if self.num_samples != 0:
            step_time = time.time()
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(
                self.batch_size_each, dim=0
            )
        #     print(f"Step 6 (stack surface points tensor) took {time.time() - step_time:.2f} seconds")

        # print(f"Total initialization time: {time.time() - start_time:.2f} seconds")

    def get_mesh(self, i):
        """
        Get object mesh with scaling applied based on precomputed scale.

        Parameters
        ----------
        i : int
            Index of the batch.

        Returns
        -------
        scaled_mesh : trimesh.base.Trimesh
            The mesh object with applied scaling.
        """
        model_index = i // self.batch_size_each
        mesh = self.object_mesh_list[model_index]
        scaled_mesh = tm.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        return scaled_mesh


    def get_plotly_data(self, i, model_scale,color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i 
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]
