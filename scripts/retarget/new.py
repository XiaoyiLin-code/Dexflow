from utils.hand_model import ManoModel
import torch
import numpy as np
import open3d as o3d
import yaml
from retarget_utils.GPU_optimizer import PositionOptimizer
from retarget_utils.robot_wrapper import RobotWrapper
import pytorch_kinematics as pk
from pathlib import Path
from utils.hand_model import create_hand_model
from config.path_config import Path_config

# 随机生成48维关节pose（不包含全局旋转）
def batch_random_hand_pose(batch_size):
    """生成批量姿态参数（形状：[batch, 48]）"""
    pose = torch.randn(batch_size, 51, device='cuda')*0.6 
    pose[:, 0:3] = 0    # 全局旋转归零
    pose[:, -3:] = 0    # 形状参数归零
    return pose[:, :51]  # 返回后48维


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
        self.cfg = self.load_yaml(self.base_dir / "config/mapping_config/shadow_hand_right.yml")
        urdf_path = "/home/lightcone/workspace/Dexflow/data/urdf/dex_retargeting/hands/shadow_hand/shadow_hand_right_glb_w_dummy.urdf"
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        self.robot = pk.build_chain_from_urdf(urdf_content).to(dtype=torch.float32, device="cuda")
        self.robot_model = create_hand_model("shadowhand", torch.device("cpu"))
        self.optimizer = PositionOptimizer(
            robot=self.robot,
            target_joint_names=self.cfg["retargeting"]["target_joint_names"],
            target_link_names=self.cfg["retargeting"]["target_link_names"],
            target_link_human_indices=self.cfg["retargeting"]["target_link_human_indices"],
            device='cuda',
            lr=0.01,
            max_iter=10,
            tol=1e-12,
        )



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


    def render_mano_data(self):
        HAND_COLOR = [0.1, 0.5, 0.9]    # 蓝色系
        
        for i in range(self.data_size):
            # 生成MANO手部网格
            mano_mesh, _ = self.hand_model.get_trans_trimesh_data(i, pose=None)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.array(mano_mesh.vertices))
            mesh.triangles = o3d.utility.Vector3iVector(np.array(mano_mesh.faces))
            mesh.compute_vertex_normals()

            # 获取并转换机器人部件
            robot_data = self.robot_model.get_transformed_links_gene_pc(
                self.qpos[i], 
                return_meshes=True
            )
            robot_meshes = convert_robot_meshes(robot_data)  # 转换为Open3D网格列表

            # 创建坐标系
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # 合并可视化对象
            visual_objects = [mesh, axis]
            visual_objects.extend(robot_meshes)
            
            # 可视化
            o3d.visualization.draw_geometries(visual_objects)

    def retarget(self):
        self.sample_mano_data()
        self.qpos=self.optimizer.optimize(self.joints, batch_size=self.joints.shape[0])
        print(self.qpos.shape)

def convert_robot_meshes(robot_data: dict) -> list:
    """将机器人部件字典转换为Open3D网格列表"""
    mesh_list = []
    ROBOT_COLOR = [0.9, 0.2, 0.1]    # 红色系
    for part_name in robot_data:
        # 提取顶点和面数据
        vert_data = robot_data[part_name]['vertices']
        face_data = robot_data[part_name]['faces']
        
        # 转换为CPU上的NumPy数组
        vertices = vert_data.cpu().numpy()  # [num_vertices, 3]
        faces = face_data.cpu().numpy()     # [num_faces, 3]
        
        # 创建Open3D网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.paint_uniform_color(ROBOT_COLOR)  # 设置机器人部件颜色
        mesh.compute_vertex_normals()
        
        mesh_list.append(mesh)

    return mesh_list
if __name__ == "__main__":
    retarget_train = Retarget_train(10000)
    retarget_train.retarget()
    retarget_train.render_mano_data()
   