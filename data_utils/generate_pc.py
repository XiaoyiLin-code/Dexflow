import os
import sys
import argparse
import time
import numpy as np
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model

# ----------------- 核心采样函数 -----------------
def farthest_point_sampling(point_cloud, num_points=1024):
    """最远点采样算法"""
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
    """生成内部点采样"""
    if num_points <= 0:
        return np.zeros((0, 3))
    
    # 处理无效网格
    if not hasattr(mesh, 'bounds') or mesh.bounds is None:
        return np.zeros((0, 3))
    
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    expanded_bounds = [
        bounds[0] - size * padding,
        bounds[1] + size * padding
    ]
    
    points = []
    while len(points) < num_points:
        samples = np.random.uniform(
            low=expanded_bounds[0],
            high=expanded_bounds[1],
            size=(num_points*2, 3)
        )
        # 加速内部点检测
        accelerator = trimesh.proximity.ProximityQuery(mesh)
        inside = accelerator.signed_distance(samples) > 0
        valid = samples[inside]
        points.extend(valid.tolist())
        if len(points) >= num_points:
            break
    
    return np.array(points[:num_points])

def sample_mesh_with_interior(mesh, num_points=1024, surface_ratio=0.5):
    """混合表面和内部点采样"""
    # 空网格处理
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        print("WARNING: Empty mesh, generating random points")
        return np.random.rand(num_points, 3), []
    
    # 表面点采样
    num_surface = max(0, min(int(num_points * surface_ratio), len(mesh.vertices)))
    surface_pts = mesh.vertices
    
    if len(surface_pts) > num_surface:
        surface_sampled, _ = farthest_point_sampling(surface_pts, num_surface)
    else:
        surface_sampled = surface_pts
    
    # 内部点采样
    num_interior = max(0, num_points - len(surface_sampled))
    interior_pts = sample_interior_points(mesh, num_interior)
    
    # 合并并调整点云
    combined = []
    if len(surface_sampled) > 0:
        combined.append(surface_sampled)
    if interior_pts.size > 0:
        combined.append(interior_pts)
    
    combined = np.vstack(combined) if combined else np.zeros((0, 3))
    
    # 保证输出点数
    if combined.shape[0] < num_points:
        padding = np.tile(combined.mean(axis=0), (num_points - combined.shape[0], 1))
        combined = np.vstack([combined, padding])
    elif combined.shape[0] > num_points:
        combined = combined[:num_points]
    
    # 最终FPS采样
    sampled, indices = farthest_point_sampling(combined, num_points)
    return sampled, indices

# ----------------- 数据生成函数 -----------------
def generate_object_pc(args):
    """生成物体点云"""
    for dataset_type in ['contactdb', 'ycb']:
        input_dir = os.path.join(ROOT_DIR, args.object_source_path, dataset_type)
        output_dir = os.path.join(ROOT_DIR, args.save_path, 'object', dataset_type)
        os.makedirs(output_dir, exist_ok=True)

        for obj_name in os.listdir(input_dir):
            obj_path = os.path.join(input_dir, obj_name)
            if not os.path.isdir(obj_path):
                continue
            
            print(f'Processing {dataset_type}/{obj_name}...')
            mesh_path = os.path.join(obj_path, f'{obj_name}.stl')
            try:
                mesh = trimesh.load_mesh(mesh_path)
            except Exception as e:
                print(f"Error loading mesh: {str(e)}")
                continue
            
            # 混合采样
            sampled_points, _ = sample_mesh_with_interior(
                mesh, 
                num_points=args.num_points,
                surface_ratio=args.surface_ratio  # 使用参数
            )
            
            # 强制形状修正
            sampled_points = np.asarray(sampled_points)
            if sampled_points.ndim != 2 or sampled_points.shape[1] != 3:
                sampled_points = sampled_points.reshape(-1, 3)[:args.num_points]
            
            # 数据清洗
            if np.isnan(sampled_points).any():
                print("WARNING: NaN detected, regenerating...")
                sampled_points = np.nan_to_num(sampled_points)
            
            # 法线计算
            dummy_normals = np.zeros(sampled_points.shape)
            accelerator = trimesh.proximity.ProximityQuery(mesh)
            for i, point in enumerate(sampled_points):
                _, dist, face_id = accelerator.on_surface([point])
                if dist[0] < 1e-6:  # 表面点
                    dummy_normals[i] = mesh.face_normals[face_id[0]]
                # 内部点保持零向量
            
            # 保存数据
            object_pc = torch.cat([
                torch.tensor(sampled_points, dtype=torch.float32),
                torch.tensor(dummy_normals, dtype=torch.float32)
            ], dim=1)
            
            torch.save(object_pc, os.path.join(output_dir, f'{obj_name}.pt'))

    print("\nObject point cloud generation completed.")

import datetime 
from collections import OrderedDict
def generate_robot_pc(args):
    output_dir = os.path.join(ROOT_DIR, args.save_path, 'robot')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.robot_name}.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hand_type="right"
    hand = create_hand_model(args.robot_name,hand_type, device, args.num_points)

    # 增强采样配置（添加容错参数）
    sampling_config = {
        'voxel_res': 1000,
        'surface_ratio': args.surface_ratio,
        'jitter_scale': 0.3,
        'multi_sample': 3,
        'monte_carlo_attempts': 5,
    }

    # try:
        # 执行增强采样（添加异常处理层）
    transformed_pc = hand.get_transformed_links_pc1(
        q=torch.zeros(hand.dof, device=device),
        internal_sample=True,
        **sampling_config
    )
    # except Exception as e:
    #     print(f"\n[Critical] Sampling failed: {str(e)}")
    #     transformed_pc = None

    # 构建兼容数据结构（修复TrackedArray问题）
    def convert_data(data):
        """通用数据转换方法"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif hasattr(data, '__array__'): 
            return np.asarray(data)
        return data

    original_data = {}
    for k, v in hand.vertices.items():
        try:
            original_data[k] = convert_data(v)
        except Exception as e:
            print(f"[WARNING] Failed to convert {k} data: {str(e)}")
            original_data[k] = np.empty((0,3))  
    filtered_data = {}
    indices=np.random.choice(transformed_pc.shape[0], args.num_points, replace=False)
    transformed_pc=transformed_pc[indices]
    if transformed_pc is not None:
        for link_index, link_name in enumerate(hand.links_pc.keys()):
            try:
                mask = transformed_pc[:, 3] == link_index
                filtered_points = transformed_pc[mask, :3]
                filtered_data[link_name] = convert_data(filtered_points)
            except Exception as e:
                print(f"[WARNING] Failed to process {link_name}: {str(e)}")
                filtered_data[link_name] = original_data.get(link_name, np.empty((0,3)))
    else:
        print("[WARNING] Using original data as fallback")
        filtered_data = original_data.copy()
    # 构建最终存储结构
    data = {
        'original': original_data,
        'filtered': filtered_data,
        'meta': {
            'robot_name': args.robot_name,
            'sampling_config': sampling_config,
            'warnings': [
                "lfdistal sampling failed",
                "thproximal sampling failed",
                "thmiddle sampling failed",
                "thdistal sampling failed"
            ]  
        }
    }

    try:
        # 转换为numpy兼容格式
        numpy_data = {
            'original': {k: v.astype(np.float32) for k, v in data['original'].items()},
            'filtered': {k: v.astype(np.float32) for k, v in data['filtered'].items()},
            'meta': data['meta']
        }
        
        torch.save(numpy_data, output_path)
        print(f"\nSaved point cloud data to {output_path}")
        print(f"Original points: {sum(len(v) for v in numpy_data['original'].values()):,}")
        print(f"Filtered points: {sum(len(v) for v in numpy_data['filtered'].values()):,}")
    except Exception as e:
        print(f"\n[Critical] Save failed: {str(e)}")
        return None

    return output_path


import importlib

def check_visualization_deps():
    """检查可视化依赖是否安装"""
    required = {'open3d', 'matplotlib'}
    missing = []
    for pkg in required:
        if not importlib.util.find_spec(pkg):
            missing.append(pkg)
    return missing

def visualize_point_cloud(file_path, point_size=2.0, window_size=(1920, 1080)):
    """通用点云可视化函数（兼容Open3D 0.12+）"""
    try:
        import open3d as o3d
        from open3d.visualization import rendering
        from matplotlib import cm
    except ImportError as e:
        print(f"无法导入可视化库: {str(e)}")
        print("请安装依赖：pip install open3d>=0.12 matplotlib")
        return False

    try:
        # 版本兼容性检测
        o3d_version = tuple(map(int, o3d.__version__.split('.')[:2]))
        text_support = o3d_version >= (0, 14)  # Text3D需要0.14+
        
        # 加载点云数据
        data = torch.load(file_path)
    except Exception as e:
        print(f"加载文件失败: {str(e)}")
        return False

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"点云预览 - {os.path.basename(file_path)}",
        width=window_size[0],
        height=window_size[1]
    )

    # 添加坐标系（所有版本支持）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)

    # 处理机器人点云数据
    if 'robot' in file_path:
        filtered_data = data.get('filtered', {})
        if not filtered_data:
            print("未找到有效的机器人点云数据")
            return False

        colormap = cm.get_cmap('gist_ncar', len(filtered_data))
        
        for idx, (link_name, points) in enumerate(filtered_data.items()):
            if len(points) == 0:
                continue

            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(colormap(idx/len(filtered_data))[:3])
            vis.add_geometry(pcd)

            # 添加文本标签（仅支持0.14+）
            if text_support:
                try:
                    centroid = np.mean(points, axis=0)
                    text = rendering.Text3D(
                        f"{link_name}", 
                        centroid, 
                        font_size=20,
                        font_family="sans-serif",
                        text_color=[1,1,1]
                    )
                    vis.add_3d_label(centroid, link_name)
                except Exception as e:
                    print(f"创建标签失败: {str(e)}")

    # 处理物体点云数据
    else:
        pc_data = data.numpy() if isinstance(data, torch.Tensor) else data
        if pc_data.shape[1] == 6:  # 带法向量
            points = pc_data[:, :3]
            normals = pc_data[:, 3:]
        else:
            points = pc_data
            normals = None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.paint_uniform_color([0.4, 0.6, 0.8])
        vis.add_geometry(pcd)

    # 配置渲染参数
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = [0.1, 0.1, 0.1]
    render_opt.light_on = True

    # 版本提示
    if 'robot' in file_path and not text_support:
        print("\n[提示] 检测到Open3D版本 < 0.14.0")
        print(" - 无法显示部件名称标签")
        print(" - 升级命令: pip install open3d>=0.14.0")

    # 运行可视化
    print("\n可视化窗口已打开... (按Q退出)")
    vis.run()
    vis.destroy_window()
    return True
# ----------------- 修改后的主程序入口 -----------------
# if __name__ == '__main__':
#     # 添加可视化参数
#     parser.add_argument('--visualize', 
#                         action='store_true',
#                         help='生成后自动可视化')
#     parser.add_argument('--point_size', 
#                         default=2.0, type=float,
#                         help='可视化点大小（像素）')
#     args = parser.parse_args()

#     # 检查可视化依赖
#     missing_deps = check_visualization_deps()
#     if args.visualize and missing_deps:
#         print(f"缺少可视化依赖: {', '.join(missing_deps)}")
#         print("请执行: pip install open3d matplotlib")
#         args.visualize = False

#     # 执行生成流程
#     start_time = time.time()
    
#     if args.type == 'robot':
#         output_path = generate_robot_pc(args)
#     else:
#         generate_object_pc(args)
#         # 物体点云需要特殊处理路径
#         dataset_type = 'contactdb' if 'contactdb' in args.object_source_path else 'ycb'
#         obj_name = os.path.basename(args.object_source_path)
#         output_path = os.path.join(
#             ROOT_DIR, args.save_path, 'object', 
#             dataset_type, f'{obj_name}.pt'
#         )

#     # 执行可视化
#     if args.visualize and output_path:
#         if os.path.exists(output_path):
#             print(f"\n正在启动可视化: {output_path}")
#             visualize_point_cloud(output_path, args.point_size)
#         else:
#             print(f"无法找到生成的文件: {output_path}")

#     # 输出耗时统计
#     elapsed = time.time() - start_time
#     print(f"\n总耗时: {elapsed:.2f}秒")
#     print("完成所有操作!")
# ----------------- 主程序入口 -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='点云生成工具')
    parser.add_argument('--type', 
                        required=True,
                        choices=['robot', 'object'],
                        help='生成类型: robot|object')
    parser.add_argument('--save_path',
                        default='data/PointCloud/',
                        type=str,
                        help='输出目录路径')
    parser.add_argument('--num_points',
                        default=5000,
                        type=int,
                        choices=range(500, 100001),
                        metavar="[500-10000]",
                        help='总采样点数')
    
    parser.add_argument('--visualize', 
                        action='store_true',
                        help='生成后自动可视化')
    parser.add_argument('--point_size', 
                        default=2.0, type=float,
                        help='可视化点大小（像素）')
    
    # 物体参数组
    object_group = parser.add_argument_group('物体选项')
    object_group.add_argument('--object_source_path',
                             default='data/data_urdf/object',
                             type=str,
                             help='源URDF目录路径')
    object_group.add_argument('--surface_ratio',
                             default=0.5,
                             type=float,
                             metavar="[0.0-1.0]",
                             help='表面点占比')
    
    # 机器人参数组
    robot_group = parser.add_argument_group('机器人选项')
    robot_group.add_argument('--robot_name',
                            default='shadow',
                            choices=['shadow', 'allegro'],
                            help='机器人模型名称')
    
    args = parser.parse_args()

    # 参数验证
    if args.type == 'object':
        if not 0.0 <= args.surface_ratio <= 1.0:
            raise ValueError("表面点比例必须在0.0到1.0之间")
        if not os.path.exists(args.object_source_path):
            raise FileNotFoundError(f"源目录不存在: {args.object_source_path}")

    # 执行生成

    if args.type == 'robot':
        output_path=generate_robot_pc(args)
    else:
        generate_object_pc(args)

    if args.visualize and output_path:
        if os.path.exists(output_path):
            print(f"\n正在启动可视化: {output_path}")
            visualize_point_cloud(output_path, args.point_size)
        else:
            print(f"无法找到生成的文件: {output_path}")

    print(f"成功生成{args.type}点云数据")
