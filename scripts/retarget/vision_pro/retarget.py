import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
def load_npy_data(file_path):
    """读取并验证实际数据结构的版本"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
            
        data = np.load(file_path, allow_pickle=False)
        
        print("数据加载成功!")
        print("数据形状:", data.shape)
        print("数据类型:", data.dtype)
        print("位置范围 X:[{:.2f}, {:.2f}] Y:[{:.2f}, {:.2f}] Z:[{:.2f}, {:.2f}]".format(
            np.min(data[..., 0]), np.max(data[..., 0]),
            np.min(data[..., 1]), np.max(data[..., 1]),
            np.min(data[..., 2]), np.max(data[..., 2]),
        ))
        
        # 验证新的数据结构
        if len(data.shape) == 3 and data.shape[1] == 26 and data.shape[2] ==7:
            print("\n数据结构验证通过:")
            print("手腕位置示例:", data[0, 0, :3])  # 假设索引0是手腕
            print("中指末端示例:", data[0, 12, :3]) # 示例中指末端
        else:
            print("\n警告：数据形状不符合单手26关节格式")
            
        return data
        
    except Exception as e:
        print("加载错误:", str(e))
        return None

def visualize_single_hand(root_state, fps=30):
    """可视化单手数据的更新版本"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)
    
    # 更新连接关系（根据26关节结构）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        # 剩余关节可能为手掌或其他部位
    ]

    def update(frame):
        ax.cla()
        ax.set_title(f"Frame: {frame}/{len(root_state)}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.5)
        ax.set_zlim(0, 1)
        
        # 提取当前帧数据
        frame_data = root_state[frame]
        
        # 绘制所有关节
        ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c='r', s=20)
        
        # 绘制连接线
        for (s, e) in connections:
            start = frame_data[s, :3]
            end = frame_data[e, :3]
            ax.plot(*zip(start, end), color='darkred', lw=1.5)
        
        # 标记手腕
        ax.scatter(*frame_data[0, :3], s=100, c='gold', marker='o')

    ani = FuncAnimation(fig, update, frames=len(root_state), interval=1000/fps)
    plt.show()

from utils.hand_model import ManoModel
import torch
import numpy as np
import open3d as o3d
import yaml
from retarget_utils.GPU_Gopt import PositionOptimizer,EnhancedPositionOptimizer
from retarget_utils.robot_wrapper import RobotWrapper
import pytorch_kinematics as pk
from pathlib import Path
from utils.hand_model import create_hand_model
from config.path_config import Path_config
from pytransform3d import rotations
from scipy.spatial.transform import Rotation
import numpy as np
def scipy_euler_to_matrix(euler_angles, order='XYZ'):
    original_shape = euler_angles.shape

    rotations = Rotation.from_euler(order, euler_angles)
    return rotations.as_matrix().reshape(*original_shape[:-1], 3, 3)


class Retarget_train:
    def __init__(self,data_size):
        self.data_size = data_size
        self.hand_model = ManoModel(
            mano_root='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano', 
            contact_indices_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano/contact_indices.json', 
            pose_distrib_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano/pose_distrib.pt', 
            device='cuda'
        )
        self.base_dir = Path(Path_config.BASE_DIR)
        self.cfg = self.load_yaml(self.base_dir / "config/mapping_config/vision_pro_allegro.yml")
        urdf_path = "/home/lightcone/workspace/Dexflow/data/urdf/dex_retargeting/hands/allegro_hand/allegro_hand_right_glb_w_dummy.urdf"
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        self.robot = pk.build_chain_from_urdf(urdf_content).to(dtype=torch.float32, device="cuda")
        self.robot_model = create_hand_model("allegro","right",torch.device("cpu"))
        self.optimizer = EnhancedPositionOptimizer(
            robot=self.robot,
            target_joint_names=self.cfg["retargeting"]["target_joint_names"],
            target_link_names=self.cfg["retargeting"]["target_link_names"],
            target_link_human_indices=self.cfg["retargeting"]["target_link_human_indices"],
            device='cuda',
            lr=0.01,
            max_iter=1000,
            tol=1e-18,
        )
        self.robot_wapper = RobotWrapper(urdf_path)

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        is_mano_convention: bool = False,
        robot=None,
        qpos=None,
    ):
        target_wrist_pose = np.eye(4)
        wrist_quat1 = wrist_quat.copy()
        wrist_quat1[0] = wrist_quat[3]
        wrist_quat1[-3:] = wrist_quat[:3]
        target_wrist_pose[:3, :3] = rotations.matrix_from_quaternion(wrist_quat1)
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
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0
        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist
        euler = rotations.euler_from_matrix(
            target_root_pose[:3, :3], 0, 1, 2, extrinsic=False
        )
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                qpos[num] = pose_vec[index]
        return qpos


    def load_yaml(self, path: Path):
            """通用的 YAML 加载方法"""
            if not path.exists():
                raise FileNotFoundError(f"配置文件未找到: {path}")
            with open(path, "r") as file:
                return yaml.safe_load(file) 
            

    def retarget(self,root_state):
        self.qpos = torch.zeros(self.data_size, len(self.cfg["retargeting"]["target_joint_names"]), 
            requires_grad=False,device="cuda").float()
        for i in range(len(root_state)):
            self.qpos[i]=self.warm_start(
                wrist_pos=root_state[i, 0, :3],
                wrist_quat=root_state[i, 0, 3:],
                robot=self.robot_wapper,
                qpos=self.qpos[0],
            )
        self.qpos=self.optimizer.optimize(torch.tensor(root_state[:,:,:3],device="cuda"),self.qpos, batch_size=len(root_state))
    def render_selected_frame(self, root_state, frame_index):
        """静态显示指定帧（显示半透明机械手网格+部件中心点）"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]

        # 初始化可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Frame {frame_index} Visualization", width=1280, height=720)
        
        # 坐标系参数设置
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis.add_geometry(coord_frame)

        # 创建关节点可视化元素
        frame_data = root_state[frame_index]
        
        # 红色关节点球体
        joint_spheres = []
        for j in range(frame_data.shape[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.paint_uniform_color([1, 0, 0])  # 纯红色
            update_geometry_position(sphere, frame_data[j, :3])
            joint_spheres.append(sphere)
            vis.add_geometry(sphere)
        
        # 金色手腕标记
        wrist_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        wrist_sphere.paint_uniform_color([1, 0.8, 0])  # 金色
        update_geometry_position(wrist_sphere, frame_data[0, :3])
        vis.add_geometry(wrist_sphere)
        
        # 深红色连接线
        for (s, e) in connections:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([frame_data[s, :3], frame_data[e, :3]])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([0.5, 0, 0])
            vis.add_geometry(line_set)
        
        # 获取机械手数据
        robot_data = self.robot_model.get_transformed_links_gene_pc(
            self.qpos[frame_index],
            return_meshes=True
        )
        
        # 同时添加半透明网格和中心点
        for i, (key, mesh_data) in enumerate(robot_data.items()):
            # ========== 半透明网格部分 ==========
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_data['vertices'].cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(mesh_data['faces'].cpu().numpy())
            mesh.paint_uniform_color([0.96, 0.93, 0.88])  # 基础颜色
            
            # 设置透明度（需要修改渲染材质）
            mesh.compute_vertex_normals()
            mesh.materials = [o3d.visualization.rendering.MaterialRecord()]
            mesh.materials[0].base_color = [0.96, 0.93, 0.88, 0.3]  # RGBA，最后一位是透明度
            mesh.materials[0].shader = "defaultLitTransparency"
            vis.add_geometry(mesh)
            
            # ========== 部件中心点 ==========
            # 计算中心点
            vertices = mesh_data['vertices'].cpu().numpy()
            center = np.mean(vertices, axis=0) if len(vertices) > 0 else np.zeros(3)
            
            # 创建蓝色小球
            part_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            part_sphere.paint_uniform_color([0.2, 0.6, 1.0])
            update_geometry_position(part_sphere, center)
            vis.add_geometry(part_sphere)

        # 视角控制参数
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, -1, 0.5])  # 俯视视角
        ctr.set_up([0, 0, 1])        # Z轴向上
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
    def render_first_frame1(self, root_state):
        """静态显示第0帧数据的专用函数"""
        # 关节连接关系定义
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]

        # 初始化可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="First Frame Visualization", width=1280, height=720)
        
        # 坐标系参数设置
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis.add_geometry(coord_frame)

        # 创建第0帧的几何元素
        frame_data = root_state[0]
        
        # 创建关节点球体
        joint_spheres = []
        for j in range(frame_data.shape[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.paint_uniform_color([1, 0, 0])
            update_geometry_position(sphere, frame_data[j, :3])
            joint_spheres.append(sphere)
            vis.add_geometry(sphere)
        
        # 创建手腕特殊标记
        wrist_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        wrist_sphere.paint_uniform_color([1, 0.8, 0])
        update_geometry_position(wrist_sphere, frame_data[0, :3])
        vis.add_geometry(wrist_sphere)
        
        # 创建连接线
        for (s, e) in connections:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([frame_data[s, :3], frame_data[e, :3]])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([0.5, 0, 0])
            vis.add_geometry(line_set)
        
        # 创建机械手模型
        robot_data = self.robot_model.get_transformed_links_gene_pc(
            self.qpos[0],  # 仅使用第0帧的qpos
            return_meshes=True
        )
        for i, (key, mesh_data) in enumerate(robot_data.items()):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_data['vertices'].cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(mesh_data['faces'].cpu().numpy())
            mesh.paint_uniform_color([0.96, 0.93, 0.88])
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
        
        # 设置静态视角参数
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, -1, 0.5])
        ctr.set_up([0, 0, 1])
        
        # 保持窗口打开
        vis.run()
        vis.destroy_window()
    def render_mano_data(self, root_state, fps=30):


        # 关节连接关系定义（与Matplotlib版本一致）
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]
        
        # 初始化可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Hand Motion Visualization", width=1280, height=720)
        
        # 坐标系参数设置
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis.add_geometry(coord_frame)

        elements = {
            'points': [],        # 关节球体
            'linesets': [],      # 连接线
            'wrist': None       # 手腕标记
        }
        
        for _ in range(root_state.shape[1]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.paint_uniform_color([1, 0, 0])  # 红色关节
            elements['points'].append(sphere)
            vis.add_geometry(sphere)
        
        # 初始化连接线
        for _ in connections:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([0.5, 0, 0])  # 深红色线条
            elements['linesets'].append(line_set)
            vis.add_geometry(line_set)
        
        # 创建手腕标记
        wrist_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        wrist_sphere.paint_uniform_color([1, 0.8, 0])  # 金色
        elements['wrist'] = wrist_sphere
        vis.add_geometry(wrist_sphere)
        
        # 设置视角参数
        ctr = vis.get_view_control()
        ctr.set_zoom(5)
        ctr.set_front([0, -1, 0.5])  # 视角方向
        ctr.set_up([0, 0, 1])        # 坐标系向上方向
        
        self.robot_meshes = []
        for _ in range(len(self.robot_model.link_names )):
            mesh = o3d.geometry.TriangleMesh()
            self.robot_meshes.append(mesh)
            vis.add_geometry(mesh)
        
        # 动画更新函数
        def update_frame(frame_idx):
            # 更新关节点位置
            print(frame_idx)
            frame_data = root_state[frame_idx]
            for j in range(frame_data.shape[0]):
                update_geometry_position(elements['points'][j], frame_data[j, :3])
                if j == 0:
                    update_geometry_position(elements['wrist'], frame_data[j, :3])
            
            # 更新机械手模型
            robot_data = self.robot_model.get_transformed_links_gene_pc(
                self.qpos[frame_idx],  # 使用当前帧的qpos
                return_meshes=True
            )
            for i, (key,mesh_data) in enumerate(robot_data.items()):
                self.robot_meshes[i].vertices = o3d.utility.Vector3dVector(mesh_data['vertices'].cpu().detach().numpy())
                self.robot_meshes[i].triangles = o3d.utility.Vector3iVector(mesh_data['faces'].cpu().detach().numpy())
                self.robot_meshes[i].paint_uniform_color([0.96, 0.93, 0.88])  # 红色
            
            # 更新连接线
            for idx, (s, e) in enumerate(connections):
                start = frame_data[s, :3]
                end = frame_data[e, :3]
                elements['linesets'][idx].points = o3d.utility.Vector3dVector([start, end])
        
        # 运行动画循环
        for frame_idx in range(len(root_state)):
            update_frame(frame_idx)
            
            # 统一更新所有几何体
            all_geoms = elements['points'] + elements['linesets'] + [elements['wrist']] + self.robot_meshes
            for geom in all_geoms:
                vis.update_geometry(geom)
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1/fps)
        
        vis.destroy_window()

        # def retarget(self):
        #     self.sample_mano_data()
        #     self.qpos = torch.zeros(self.data_size, len(self.cfg["retargeting"]["target_joint_names"]), 
        #             requires_grad=False,device="cuda").float()
        #     # self.qpos=self.warm_start(
        #     #     wrist_pos=self.joints[:,0,:].cpu().numpy(),
        #     #     wrist_euler=self.full_pose[:, :3].cpu().numpy(),
        #     #     qpos=self.qpos,
        #     # )
        #     for i in range(self.data_size):
        #         self.qpos[i]=self.warm_start(
        #             wrist_pos=self.joints[i,0,:].cpu().numpy(),
        #             wrist_euler=self.full_pose[i, :3].cpu().numpy(),
        #             robot=self.robot_wapper,
        #             qpos=self.qpos[0],
        #         )
        #     init_6d=self.qpos[:,:6].clone()

        #     self.qpos=self.optimizer.optimize(self.joints,self.qpos, batch_size=self.joints.shape[0])
        #     self.qpos=torch.cat([init_6d,self.qpos],dim=1)

def convert_robot_meshes(robot_data: dict) -> list:
    """将机器人部件字典转换为Open3D网格列表"""
    mesh_list = []
    ROBOT_COLOR = [0.96, 0.93, 0.88]    # 红色系
    for part_name in robot_data:
        # 提取顶点和面数据
        vert_data = robot_data[part_name]['vertices']
        face_data = robot_data[part_name]['faces']
        
        # 转换为CPU上的NumPy数组
        vertices = vert_data.cpu().detach().numpy()  # [num_vertices, 3]
        faces = face_data.cpu().detach().numpy()     # [num_faces, 3]
        
        # 创建Open3D网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        if part_name == 'link_14.0':
            mesh.paint_uniform_color([1, 1, 1])
        else:
            mesh.paint_uniform_color(ROBOT_COLOR)  # 设置机器人部件颜色
        mesh.compute_vertex_normals()
        mesh_list.append(mesh)

    return mesh_list

def update_geometry_position(geom, new_pos):
    # 重置到原点
    current_center = geom.get_center()
    reset_transform = np.eye(4)
    reset_transform[:3, 3] = -current_center
    geom.transform(reset_transform)
    
    # 应用新位置
    new_transform = np.eye(4)
    new_transform[:3, 3] = new_pos
    geom.transform(new_transform)




if __name__ == "__main__":
    # 配置路径
    data_dir = "/home/lightcone/workspace/Dexflow/scripts/retarget/vision_pro/traj_data/"
    file_name = "all_frames1.npy"
    file_path = os.path.join(data_dir, file_name)
    
    # 加载数据
    motion_data = load_npy_data(file_path)
    motion_data[..., 2] -= 1  # Y轴偏移示例
    motion_data[..., :]*=1.50
    # motion_data[:,2,:]=0
    print(motion_data.shape)
    retarget_train = Retarget_train(len(motion_data))
    if motion_data is not None:
        # 数据预处理
        motion_data = motion_data.astype(np.float32)
        retarget_train.retarget(motion_data)
        while 1:
            retarget_train.render_mano_data(motion_data[:],120)
            # retarget_train.render_selected_frame(motion_data[:],480)