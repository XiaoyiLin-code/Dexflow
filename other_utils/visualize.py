import os
import torch
import numpy as np
import open3d as o3d
import json
import yaml
from collections.abc import Mapping, Iterable

import plotly.graph_objects as go

def rpy_to_normal(roll, pitch, yaw):
    """
    将欧拉角（Roll, Pitch, Yaw）转换为法向量（Z 轴方向）
    :param roll: 绕 X 轴的旋转角（弧度）
    :param pitch: 绕 Y 轴的旋转角（弧度）
    :param yaw: 绕 Z 轴的旋转角（弧度）
    :return: 单位法向量 (shape: (3,))
    """
    # 绕 X 轴的旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 绕 Y 轴的旋转矩阵
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # 绕 Z 轴的旋转矩阵
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    world_z = np.array([0, 0, 1])
    # 合成总旋转矩阵：R = R_z @ R_y @ R_x
    R = R_z @ R_y @ R_x @ world_z

    # 提取第三列（Z 轴方向）
    normal = R
    return normal

# 示例用法
roll = np.radians(30)   # 30 度转为弧度
pitch = np.radians(45)
yaw = np.radians(60)
normal = rpy_to_normal(roll, pitch, yaw)
print("法向量:", normal)

def euler_rotation_matrix(roll, pitch, yaw):
    """计算XYZ顺序的欧拉角旋转矩阵 (roll-X, pitch-Y, yaw-Z)"""
    # 绕X轴的旋转（roll）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    # 绕Y轴的旋转（pitch）
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # 绕Z轴的旋转（yaw）
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # 组合旋转顺序：Z -> Y -> X（注意矩阵右乘顺序）
    return Rz @ Ry @ Rx


def visualize_pose(fig,pose, rot):
    # 欧拉角转换为弧度（若需要）
    roll, pitch, yaw = rot[0], rot[1], rot[2]
    R = euler_rotation_matrix(roll, pitch, yaw)
    
    # 提取旋转后的坐标轴方向
    x_dir = R[:, 0]
    y_dir = R[:, 1]
    z_dir = R[:, 2]
    position = np.array(pose)
    
    # 创建3D图形

    
    # 绘制位置点
    fig.add_trace(go.Scatter3d(
        x=[position[0]], y=[position[1]], z=[position[2]],
        mode='markers',
        marker=dict(size=5, color='black')
    ))
    
    # 坐标轴参数
    arrow_length = 1.0  # 箭头长度
    cone_ratio = 0.1    # 锥体比例
    arrow_l=0
    # 绘制X轴（红色）
    fig.add_trace(go.Cone(
        x=[position[0] + x_dir[0]*arrow_l],
        y=[position[1] + x_dir[1]*arrow_l],
        z=[position[2] + x_dir[2]*arrow_l],
        u=[x_dir[0]*arrow_length],  # 方向向量
        v=[x_dir[1]*arrow_length],
        w=[x_dir[2]*arrow_length],
        sizemode="scaled",
        sizeref=cone_ratio,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False
    ))
    
    # 绘制Y轴（绿色）
    fig.add_trace(go.Cone(
        x=[position[0] + y_dir[0]*arrow_l],
        y=[position[1] + y_dir[1]*arrow_l],
        z=[position[2] + y_dir[2]*arrow_l],
        u=[y_dir[0]*arrow_length],
        v=[y_dir[1]*arrow_length],
        w=[y_dir[2]*arrow_length],
        sizemode="scaled",
        sizeref=cone_ratio,
        colorscale=[[0, 'green'], [1, 'green']],
        showscale=False
    ))
    
    # 绘制Z轴（蓝色）
    fig.add_trace(go.Cone(
        x=[position[0] + z_dir[0]*arrow_l],
        y=[position[1] + z_dir[1]*arrow_l],
        z=[position[2] + z_dir[2]*arrow_l],
        u=[z_dir[0]*arrow_length],
        v=[z_dir[1]*arrow_length],
        w=[z_dir[2]*arrow_length],
        sizemode="scaled",
        sizeref=cone_ratio,
        colorscale=[[0, 'blue'], [1, 'blue']],
        showscale=False
    ))
    
    # 设置3D场景布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(200,200,230,0.5)"),
            yaxis=dict(backgroundcolor="rgba(230,200,230,0.5)"),
            zaxis=dict(backgroundcolor="rgba(230,230,200,0.5)"),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
def visualize_and_save_transformed_links_and_object_pc(
    robot_pc=None,
    robot_pc_origin=None,
    robot_mesh=None,
    object_pc=None,
    mano_mesh=None,
    object_mesh=None,
    keypoints=None,
    save_path=None,
    contact_point_indices=None,
    mano_contact_point_indices=None,
    available_indices=None,
    object_contact_point_indices=None,
    vector=None,
    vector2=None,
    vector3=None,
    vector4=None,
    
):

    fig = go.Figure()
    if vector is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, vector[0]],
            y=[0, vector[1]],
            z=[0, vector[2]],
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=4),
            name='Vector'
        ))

        # 添加箭头锥体（可选美化）
        fig.add_trace(go.Cone(
            x=[vector[0]],
            y=[vector[1]],
            z=[vector[2]],
            u=[vector[0]],
            v=[vector[1]],
            w=[vector[2]],
            sizemode="absolute",
            sizeref=0.2,
            anchor="tip",
            showscale=False,
            colorscale="Blues",
            name="Arrowhead"
        ))
    if vector2 is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, vector2[0]],
            y=[0, vector2[1]],
            z=[0, vector2[2]],
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=4),
            name='Vector2'
        ))

        # 添加箭头锥体（可选美化）
        fig.add_trace(go.Cone(
            x=[vector2[0]],
            y=[vector2[1]],
            z=[vector2[2]],
            u=[vector2[0]],
            v=[vector2[1]],
            w=[vector2[2]],
            sizemode="absolute",
            sizeref=0.2,
            anchor="tip",
            showscale=False,
            colorscale="Reds",
            name="Arrowhead"
        ))
    if vector3 is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, vector3[0]],
            y=[0, vector3[1]],
            z=[0, vector3[2]],
            mode='lines+markers',
            line=dict(color='green', width=6),
            marker=dict(size=4),
            name='Vector3'
        ))

        # 添加箭头锥体（可选美化）
        fig.add_trace(go.Cone(
            x=[vector3[0]],
            y=[vector3[1]],
            z=[vector3[2]],
            u=[vector3[0]],
            v=[vector3[1]],
            w=[vector3[2]],
            sizemode="absolute",
            sizeref=0.2,
            anchor="tip",
            showscale=False,
            colorscale="Greens",
            name="Arrowhead"
        ))
    if vector4 is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, vector4[0]],
            y=[0, vector4[1]],
            z=[0, vector4[2]],
            mode='lines+markers',
            line=dict(color='orange', width=6),
            marker=dict(size=4),
            name='Vector4'
        ))

        # 添加箭头锥体（可选美化）
        fig.add_trace(go.Cone(
            x=[vector4[0]],
            y=[vector4[1]],
            z=[vector4[2]],
            u=[vector4[0]],
            v=[vector4[1]],
            w=[vector4[2]],
            sizemode="absolute",
            sizeref=0.2,
            anchor="tip",
            showscale=False,
            colorscale="Oranges",
            name="Arrowhead"
        ))

    if robot_pc is not None:

        robot_pc = robot_pc.detach().cpu().numpy()
        robot_x, robot_y, robot_z = robot_pc[:, 0], robot_pc[:, 1], robot_pc[:, 2]
        robot_link_indices = robot_pc[:, 3]
        # robot_x[:]+=0.15

        unique_links = sorted(set(robot_link_indices))
        robot_colors = [
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "cyan",
            "pink",
            "yellow",
        ]

        for i, link_index in enumerate(unique_links):
            link_mask = robot_link_indices == link_index  
            link_x = robot_x[link_mask]
            link_y = robot_y[link_mask]
            link_z = robot_z[link_mask]
            fig.add_trace(
                go.Scatter3d(
                    x=link_x,
                    y=link_y,
                    z=link_z,
                    mode="markers",
                    marker=dict(size=2, color=robot_colors[i % len(robot_colors)]),
                    name=f"Robot Link {int(link_index)}",
                )
            )
        if contact_point_indices is not None:
            print(contact_point_indices)
            if isinstance(contact_point_indices, torch.Tensor):
                contact_point_indices = contact_point_indices.cpu().numpy()
            contact_x = robot_x[contact_point_indices]
            contact_y = robot_y[contact_point_indices]
            contact_z = robot_z[contact_point_indices]
            fig.add_trace(
                go.Scatter3d(
                    x=contact_x,
                    y=contact_y,
                    z=contact_z,
                    mode="markers",
                    marker=dict(size=15, color="gold", symbol="cross"),
                    name="Robot Contact Points",
                )
            )

    if robot_pc_origin is not None:
        robot_pc_origin = robot_pc_origin.detach().cpu().numpy()
        origin_x, origin_y, origin_z = (
            robot_pc_origin[:, 0],
            robot_pc_origin[:, 1],
            robot_pc_origin[:, 2],
        )
        fig.add_trace(
            go.Scatter3d(
                x=origin_x,
                y=origin_y,
                z=origin_z,
                mode="markers",
                marker=dict(size=2, color="gray", symbol="circle"),
                name="Robot PC Origin",
            )
        )

    if available_indices is not None:
        if isinstance(available_indices, torch.Tensor):
            available_indices = available_indices.cpu().numpy()
        contact_x = robot_x[available_indices]
        contact_y = robot_y[available_indices]
        contact_z = robot_z[available_indices]
        fig.add_trace(
            go.Scatter3d(
                x=contact_x,
                y=contact_y,
                z=contact_z,
                mode="markers",
                marker=dict(size=5, color="black", symbol="diamond"),
                name="Robot Contact Points",
            )
        )

    if object_pc is not None:
        object_pc = object_pc.detach().cpu().numpy()
        object_x, object_y, object_z = object_pc[:, 0], object_pc[:, 1], object_pc[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=object_x,
                y=object_y,
                z=object_z,
                mode="markers",
                marker=dict(size=3, color="white"),
                name="Object Point Cloud",
            )
        )
    # object_contact_point_indices=object_contact_point_indices.squeeze(0)
    if object_contact_point_indices is not None:
        if isinstance(object_contact_point_indices, torch.Tensor):
            object_contact_point_indices = object_contact_point_indices.cpu().numpy()
        object_contact_x = object_x[object_contact_point_indices]
        object_contact_y = object_y[object_contact_point_indices]
        object_contact_z = object_z[object_contact_point_indices]
        fig.add_trace(
            go.Scatter3d(
                x=object_contact_x,
                y=object_contact_y,
                z=object_contact_z,
                mode="markers",
                marker=dict(size=5, color="red", symbol="circle"),
                name="Object Contact Points",
            )
        )
    def ensure_numpy(data):
            if isinstance(data, tuple):
                return data[1]  
            elif isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            return data
    if robot_mesh is not None:
        for mesh_name in robot_mesh:
            mesh_data = robot_mesh[mesh_name]
            vertices = ensure_numpy(mesh_data["vertices"])
            faces = ensure_numpy(mesh_data["faces"])
            vertices = vertices.astype(np.float32)
            faces = faces.astype(np.int32)
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.5,
                    color="black",
                    name=f"Robot Mesh {mesh_name}",
                )
            )
    

    if mano_mesh is not None:

        # normal_robot=rpy_to_normal(
        #     roll=rpy[0],
        #     pitch=rpy[1],
        #     yaw=rpy[2]
        # )
        # fig.add_trace(
        #     go.Cone(
        #         x=[origin[0]],
        #         y=[origin[1]],
        #         z=[origin[2]],
        #         u=[normal_robot[0]],
        #         v=[normal_robot[1]],
        #         w=[normal_robot[2]],
        #         sizemode="scaled",
        #         sizeref=0.2,
        #         colorscale=[[0, "blue"]],
        #         showscale=False
        #     )
        # )

        if isinstance(mano_mesh, torch.Tensor):
            mano_vertices = mano_mesh.detach().cpu().numpy()
        elif isinstance(mano_mesh, np.ndarray):
            mano_vertices = mano_mesh
        else:
            mano_vertices = mano_mesh.vertices
        mano_x, mano_y, mano_z = (
            mano_vertices[:, 0],
            mano_vertices[:, 1],
            mano_vertices[:, 2],
        )
        fig.add_trace(
            go.Scatter3d(
                x=mano_x,
                y=mano_y,
                z=mano_z,
                mode="markers",
                marker=dict(size=2, color="brown"),
                name="MANO Mesh",
            )
        )
        
        #找到x最小的15个点
#         min_x_indices = np.argsort(mano_x)[:100]
#         face_indices = [38, 122, 118, 117, 119, 120, 108, 79, 78, 121, 214, 215, 279, 239, 234, 92]
#         face_vertices = mano_vertices[face_indices]  # 形状为 (16, 3)
#         centroid = np.mean(face_vertices, axis=0)
#         points_centered = face_vertices - centroid
#         U, S, Vt = np.linalg.svd(points_centered)
#         normal = Vt[2]  # 第三行对应最小奇异值的方向（平面法向量）

#         normal = normal / np.linalg.norm(normal)
#         fig.add_trace(
#             go.Cone(
#                 x=[centroid[0]],  # 箭头起点 x（质心坐标）
#                 y=[centroid[1]],  # 箭头起点 y
#                 z=[centroid[2]],  # 箭头起点 z
#                 u=[normal[0]],    # 法向量方向 x 分量
#                 v=[normal[1]],    # 法向量方向 y 分量
#                 w=[normal[2]],    # 法向量方向 z 分量
#                 sizemode="scaled",# 缩放模式
#                 sizeref=0.1,      # 控制箭头大小（值越小箭头越长）
#                 colorscale=[[0, "red"], [1, "red"]],  # 纯红色
#                 showscale=False,  # 不显示颜色条
#                 name="Face Normal"
#             )
#         )

#         # 可选：添加质心标记（蓝色点）
#         fig.add_trace(
#             go.Scatter3d(
#                 x=[centroid[0]],
#                 y=[centroid[1]],
#                 z=[centroid[2]],
#                 mode="markers",
#                 marker=dict(size=5, color="blue"),
#                 name="Centroid"
#             )
# )
        # for min_x_indice in face_indices:
        #     ax = [mano_x[min_x_indice]]
        #     ay = [mano_y[min_x_indice]]
        #     az = [mano_z[min_x_indice]]
        #     fig.add_trace(
        #             go.Scatter3d(
        #                 x=ax,
        #                 y=ay,
        #                 z=az,
        #                 mode="markers",
        #                 marker=dict(size=10, color="green", symbol="circle"),
        #                 name=f"MANO,{min_x_indice}",
        #             )
        #     )
        if mano_contact_point_indices is not None:
            if isinstance(mano_contact_point_indices, torch.Tensor):
                mano_contact_point_indices = mano_contact_point_indices.cpu().numpy()
            mano_contact_x = mano_x[mano_contact_point_indices]
            mano_contact_y = mano_y[mano_contact_point_indices]
            mano_contact_z = mano_z[mano_contact_point_indices]

            fig.add_trace(
                go.Scatter3d(
                    x=mano_contact_x,
                    y=mano_contact_y,
                    z=mano_contact_z,
                    mode="markers",
                    marker=dict(size=8, color="lime", symbol="circle"),
                    name="MANO Contact Points",
                )
            )

    if object_mesh is not None:
        object_vertices = object_mesh.vertices
        object_faces = object_mesh.faces
        object_x, object_y, object_z = (
            object_vertices[:, 0],
            object_vertices[:, 1],
            object_vertices[:, 2],
        )
        fig.add_trace(
            go.Mesh3d(
                x=object_x,
                y=object_y,
                z=object_z,
                i=object_faces[:, 0],
                j=object_faces[:, 1],
                k=object_faces[:, 2],
                opacity=0.1,
                color="lightblue",
                name="Object Mesh",
            )
        )

    if keypoints is not None:
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        keypoint_x, keypoint_y, keypoint_z = (
            keypoints[:, 0],
            keypoints[:, 1],
            keypoints[:, 2],
        )
        fig.add_trace(
            go.Scatter3d(
                x=keypoint_x,
                y=keypoint_y,
                z=keypoint_z,
                mode="markers",
                marker=dict(size=5, color="gold", symbol="diamond"),
                name="Keypoints",
            )
        )

    fig.update_layout(
        title="Robot, Object, MANO Mesh, and Contact Points Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    fig.show()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_pc[:, :3])
        o3d.io.write_point_cloud(os.path.join(save_path, "robot_pc.ply"), robot_pcd)
        if object_pc is not None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_pc)
            o3d.io.write_point_cloud(
                os.path.join(save_path, "object_pc.ply"), object_pcd
            )
        if mano_mesh is not None:
            mano_pcd = o3d.geometry.PointCloud()
            mano_pcd.points = o3d.utility.Vector3dVector(mano_vertices)
            o3d.io.write_point_cloud(os.path.join(save_path, "mano_mesh.ply"), mano_pcd)

def tensor_to_list(obj):    
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return type(obj)({k: tensor_to_list(v) for k, v in obj.items()})
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        container_type = type(obj)
        if container_type in (list, tuple, set, frozenset):
            return container_type(tensor_to_list(v) for v in obj)
        return [tensor_to_list(v) for v in obj]
    return obj

def save_object_name_as_json(dataset, output_file):
    object_names = []

    for data in dataset:
        object_name = data.get("object_name") 
        if object_name:
            object_names.append(object_name)

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(object_names, json_file, ensure_ascii=False, indent=4)

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data
