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
# 随机生成48维关节pose（不包含全局旋转）
def batch_random_hand_pose(batch_size):
    """生成批量姿态参数（形状：[batch, 48]）"""
    pose = torch.randn(batch_size, 51, device='cuda')*0.7
    pose[:, :3] = 0    # 全局旋转归零
    pose[:, -3:] = 0    # 形状参数归零
    return pose[:, :51]  # 返回后48维

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
        self.cfg = self.load_yaml(self.base_dir / "config/mapping_config/allegro_hand_right.yml")
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
            max_iter=1500,
            tol=1e-18,
        )
        self.robot_wapper = RobotWrapper(urdf_path)

    def warm_start_old1(
        self,
        wrist_pos: np.ndarray,
        wrist_euler: np.ndarray,
        is_mano_convention: bool = False,
        robot=None,
        qpos=None,
    ):
        target_wrist_pose = np.eye(4)
        wrist_quat=rotations.quaternion_from_compact_axis_angle(wrist_euler)
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

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_euler: np.ndarray,
        is_mano_convention: bool = False,
        qpos=None,
    ):
        robot = self.optimizer.robot
        target_wrist_pose = np.zeros((len(wrist_euler), 4, 4))  # 形状 (N, 3, 3)
        print(target_wrist_pose.shape)
        target_wrist_pose[:,:3,:3] =  Rotation.from_euler("XYZ", wrist_euler).as_matrix()
        target_wrist_pose[:, 3, 3] = 1.0 
        target_wrist_pose[:,:3, 3] = wrist_pos
        target_wrist_pose=torch.tensor(target_wrist_pose,device="cpu",dtype=torch.float32).detach()
        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        print(dir(robot))   # 列出所有属性和方法
    

        link_pose=robot.forward_kinematics(self.qpos)["base_link"].get_matrix()
        inv_pose = torch.stack([torch.linalg.inv(p) for p in link_pose]).detach().cpu()
        target_root_pose = target_wrist_pose @ inv_pose
        euler = torch.tensor(Rotation.from_matrix(target_root_pose[:,:3,:3]).as_euler("XYZ",degrees=False))
        pose_vec = torch.cat([
            target_root_pose[:,:3, 3],  # 平移部分
            euler                # 欧拉角
        ], dim=-1)  # 沿最后一个维度拼接
        qpos=qpos.detach()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                qpos[:, num] = pose_vec[:, index].clone().detach()
                print( pose_vec[:, index],joint_name)

        return qpos

    def load_yaml(self, path: Path):
        """通用的 YAML 加载方法"""
        if not path.exists():
            raise FileNotFoundError(f"配置文件未找到: {path}")
        with open(path, "r") as file:
            return yaml.safe_load(file) 

    def sample_mano_data(self):
        random_pose_48 = batch_random_hand_pose(self.data_size)
        full_pose = random_pose_48.to('cuda')
        self.vertices,self.joints,_=self.hand_model.set_parameters(full_pose, data_origin="ycb")
        self.vertices*=10/6
        self.joints*=10/6
        self.full_pose = full_pose


    def render_mano_data(self):
        HAND_COLOR = [0.1, 0.5, 0.9]  # 蓝色系
        JOINT_COLOR = [0, 1, 0]       # 绿色表示关节点
        
        for i in range(self.data_size):
            # 生成MANO手部网格
            mano_mesh, _ = self.hand_model.get_trans_trimesh_data(i, pose=None)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.array(mano_mesh.vertices))
            mesh.triangles = o3d.utility.Vector3iVector(np.array(mano_mesh.faces))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(HAND_COLOR)

            # 获取并转换机器人部件
            robot_data = self.robot_model.get_transformed_links_gene_pc(
                self.qpos[i], 
                return_meshes=True
            )
            robot_meshes = convert_robot_meshes(robot_data) 

            # 创建关节点球体
            joints_i = self.joints[i].detach().cpu().numpy()  # [num_joints, 3]
            joint_spheres = []
            for j,point in enumerate(joints_i):
                if j!=2 and j!=3 and j!=4:
                    continue
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                sphere.translate(point)
                sphere.paint_uniform_color(JOINT_COLOR)
                joint_spheres.append(sphere)

            # 创建坐标系
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # 合并可视化对象
            visual_objects = [mesh,axis]
            visual_objects.extend(robot_meshes)
            visual_objects.extend(joint_spheres)  # 添加关节点
            
            # 可视化
            o3d.visualization.draw_geometries(visual_objects)

    def retarget(self):
        self.sample_mano_data()
        self.qpos = torch.zeros(self.data_size, len(self.cfg["retargeting"]["target_joint_names"]), 
                 requires_grad=False,device="cuda").float()
        # self.qpos=self.warm_start(
        #     wrist_pos=self.joints[:,0,:].cpu().numpy(),
        #     wrist_euler=self.full_pose[:, :3].cpu().numpy(),
        #     qpos=self.qpos,
        # )
        for i in range(self.data_size):
            self.qpos[i]=self.warm_start_old1(
                wrist_pos=self.joints[i,0,:].cpu().numpy(),
                wrist_euler=self.full_pose[i, :3].cpu().numpy(),
                robot=self.robot_wapper,
                qpos=self.qpos[0],
            )
        init_6d=self.qpos[:,:6].clone()

        self.qpos=self.optimizer.optimize(self.joints,self.qpos, batch_size=self.joints.shape[0])
        self.qpos=torch.cat([init_6d,self.qpos],dim=1)
def convert_robot_meshes(robot_data: dict) -> list:
    """将机器人部件字典转换为Open3D网格列表"""
    mesh_list = []
    ROBOT_COLOR = [0.9, 0.2, 0.1]    # 红色系
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
if __name__ == "__main__":
    retarget_train = Retarget_train(10000)
    retarget_train.retarget()
    retarget_train.render_mano_data()
   