import os
import sys
import torch
import numpy as np
import trimesh


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def calculate_depth(robot_pc, object_names):
    """
    Calculate the average penetration depth of predicted pc into the object.

    :param robot_pc: (B, N, 3)
    :param object_name: list<str>, len = B
    :return: calculated depth, (B,)
    """
    object_pc_list = []
    normals_list = []
    for object_name in object_names:
        name = object_name.split('+')
        object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
        object_pc_normals = torch.load(object_path).to(robot_pc.device)
        object_pc_list.append(object_pc_normals[:, :3])
        normals_list.append(object_pc_normals[:, 3:])
    object_pc = torch.stack(object_pc_list, dim=0)
    normals = torch.stack(normals_list, dim=0)

    distance = torch.cdist(robot_pc, object_pc)
    distance, index = torch.min(distance, dim=-1)
    index = index.unsqueeze(-1).repeat(1, 1, 3)
    object_pc_indexed = torch.gather(object_pc, dim=1, index=index)
    normals_indexed = torch.gather(normals, dim=1, index=index)
    get_sign = torch.vmap(torch.vmap(lambda x, y: torch.where(torch.dot(x, y) >= 0, 1, -1)))
    signed_distance = distance * get_sign(robot_pc - object_pc_indexed, normals_indexed)
    signed_distance[signed_distance > 0] = 0
    return -torch.mean(signed_distance)


def farthest_point_sampling(point_cloud, num_points=1024):
    """
    :param point_cloud: (N, 3) or (N, 4), point cloud (with link index)
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 4), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] == 4:
        point_cloud = point_cloud[:, :3]

    selected_indices = [0]
    distances = torch.norm(point_cloud - point_cloud[selected_indices[-1]], dim=1)
    for _ in range(num_points - 1):
        farthest_point_idx = torch.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = torch.norm(point_cloud - point_cloud[farthest_point_idx], dim=1)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    return sampled_point_cloud, selected_indices

def farthest_point_sampling(point_cloud, num_points=1024):
    """
    最远点采样算法，支持带有额外属性的点云
    :param point_cloud: (N, 3+) numpy数组或torch张量
    :param num_points: 需要采样的点数
    :return: 采样后的点云和索引
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_numpy = isinstance(point_cloud, np.ndarray)
    
    if is_numpy:
        point_cloud = torch.from_numpy(point_cloud).float().to(device)
    
    origin_shape = point_cloud.shape
    pc_xyz = point_cloud[:, :3] if origin_shape[1] > 3 else point_cloud
    
    N = pc_xyz.shape[0]
    if N == 0:
        return (np.empty((0, *origin_shape[1:])) if is_numpy 
                else torch.empty((0, *origin_shape[1:]), device=device)), []
    
    selected = [0]
    distances = torch.norm(pc_xyz - pc_xyz[selected[-1]], dim=1)
    
    for _ in range(1, num_points):
        if len(selected) >= N:
            break
        farthest = torch.argmax(distances).item()
        selected.append(farthest)
        new_dist = torch.norm(pc_xyz - pc_xyz[farthest], dim=1)
        distances = torch.min(distances, new_dist)
    
    sampled = point_cloud[selected]
    return (sampled.cpu().numpy(), selected) if is_numpy else (sampled, selected)

def sample_interior_points(mesh, num_points, padding=0.05):
    """
    在网格包围盒内生成内部点
    :param mesh: trimesh网格对象
    :param num_points: 需要生成的内部点数
    :param padding: 包围盒扩展比例
    :return: 内部点坐标数组
    """
    if num_points <= 0:
        return np.zeros((0, 3))
    
    bounds = mesh.bounds
    if bounds is None:
        return np.zeros((0, 3))
    
    # 扩展包围盒
    size = bounds[1] - bounds[0]
    expanded_bounds = [
        bounds[0] - size * padding,
        bounds[1] + size * padding
    ]
    
    points = []
    while len(points) < num_points:
        # 生成候选点
        samples = np.random.uniform(
            low=expanded_bounds[0],
            high=expanded_bounds[1],
            size=(num_points*2, 3)
        )
        # 精确的体素筛选
        inside = mesh.contains(samples)
        valid = samples[inside]
        points.extend(valid.tolist())
        if len(points) >= num_points:
            break
    
    return np.array(points[:num_points])

def sample_mesh_with_interior(mesh, num_points=1024, surface_ratio=0.5):
    """
    网格采样主函数
    :param mesh: trimesh网格对象
    :param num_points: 总采样点数
    :param surface_ratio: 表面点占比
    :return: 采样后的点云和索引
    """
    # 表面点采样
    num_surface = int(num_points * surface_ratio)
    surface_pts = mesh.vertices
    
    if len(surface_pts) > num_surface:
        surface_sampled, _ = farthest_point_sampling(surface_pts, num_surface)
    else:
        surface_sampled = surface_pts
    
    # 内部点采样
    num_interior = num_points - num_surface
    interior_pts = sample_interior_points(mesh, num_interior)
    
    # 合并点云
    combined = np.vstack([surface_sampled, interior_pts]) if interior_pts.size > 0 else surface_sampled
    
    # 最终FPS采样
    sampled, indices = farthest_point_sampling(combined, num_points)
    return sampled, indices