from utils.hand_model import ManoModel
import torch
import numpy as np
import open3d as o3d
import yaml
from retarget_utils.GPU_Gopt import PositionOptimizer
from retarget_utils.robot_wrapper import RobotWrapper
import pytorch_kinematics as pk
from pathlib import Path
from utils.hand_model import create_hand_model
from config.path_config import Path_config
from torch.utils.data import DataLoader
from scripts.retarget.nn.RetargetNet import FingerLSTM, RetargetDataset,FingerNet
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pytransform3d import rotations
# 随机生成48维关节pose（不包含全局旋转）
def batch_random_hand_pose(batch_size):
    """生成批量姿态参数（形状：[batch, 48]）"""
    pose = torch.randn(batch_size, 51, device='cuda')*0.8
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
        print(dir(self.robot),self.robot.get_joints())  # 返回所有属性和方法的列表
        self.robot_model = create_hand_model("shadow","right",torch.device("cpu"))
        self.optimizer = PositionOptimizer(
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
        self.vertices*=10/7
        self.joints*=10/7
        self.full_pose = full_pose


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
            print(self.full_qpos[i],self.robot_model.dof_name,self.robot)
            robot_data = self.robot_model.get_transformed_links_gene_pc(
                self.full_qpos[i], 
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
        self.qpos = torch.zeros(self.data_size,28, 
                 requires_grad=False,device="cuda").float()
        # self.qpos=self.warm_start(
        #     wrist_pos=self.joints[:,0,:].cpu().numpy(),
        #     wrist_euler=self.full_pose[:, :3].cpu().numpy(),
        #     qpos=self.qpos,
        # )
        self.full_qpos = self.qpos.clone()
        for i in range(self.data_size):
            self.qpos[i]=self.warm_start_old1(
                wrist_pos=self.joints[i,0,:].cpu().numpy(),
                wrist_euler=self.full_pose[i, :3].cpu().numpy(),
                robot=self.robot_wapper,
                qpos=self.qpos[0],
            )
        self.qpos=self.optimizer.optimize(self.joints, batch_size=self.joints.shape[0])
        print(self.qpos.shape)
        self.full_qpos=self.qpos
        self.render_mano_data()

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
        mesh.paint_uniform_color(ROBOT_COLOR)  # 设置机器人部件颜色
        mesh.compute_vertex_normals()
        
        mesh_list.append(mesh)

    return mesh_list


def main():
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # 数据生成
    print("Generating training data...")

    retarget = Retarget_train(20000)  # 增大数据量
    retarget.retarget()
    
    # 创建数据集
    dataset = RetargetDataset(
        retarget.joints.cpu().float(),
        retarget.qpos.cpu().float()
    )
    dataset_path = Path("saved_datasets/training_data.pt")
    if not dataset_path.parent.exists():
        dataset_path.parent.mkdir(parents=True)

    # 保存训练数据
    torch.save({
        'joints': retarget.joints.cpu(),
        'qpos': retarget.qpos.cpu(),
        'joints_mean': dataset.joints_mean,
        'joints_std': dataset.joints_std,
        'qpos_min': dataset.qpos_min,
        'qpos_max': dataset.qpos_max,
    }, dataset_path)

    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # 模型初始化
    model = FingerLSTM(
        lower=retarget.optimizer.lower_limits,
        upper=retarget.optimizer.upper_limits
    ).to(device)
    
    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-5
    )
    
    # 训练参数
    best_val_loss = float('inf')
    early_stop_counter = 0
    epochs = 200
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # 混合精度前向
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    print("out",outputs.shape,targets.shape)
                    loss = F.mse_loss(outputs, targets[:,6:10])
                    # loss1=F.mse_loss(outputs[:6], targets[:,:6])
                # loss+=loss1
                # 反向传播优化
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # 更新进度条
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        
        # 验证阶段
        model.eval()
        val_inputs, val_targets = dataset.validation_set
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            val_outputs = model(val_inputs)
            val_loss = F.mse_loss(val_outputs, val_targets[:,6:10]).item()
            # val_loss1 = F.mse_loss(val_outputs[:6], val_targets[:,:6]).item()
            # val_loss+=val_loss1
        
        # 学习率调整
        scheduler.step()
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save({
                "model": model.state_dict(),
                "joints_mean": dataset.joints_mean,
                "joints_std": dataset.joints_std,
                "qpos_min": dataset.qpos_min,
                "qpos_max": dataset.qpos_max,
            }, "best_model.pth")
        else:
            early_stop_counter += 1
            
        # 打印训练信息
        lr = optimizer.param_groups[0]["lr"]
        print(f"Val Loss: {val_loss:.4e} | LR: {lr:.2e} | Best: {best_val_loss:.4e}")
        
        # 早停判断
        if early_stop_counter >= 30:
            print("Early stopping triggered")
            break
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint["model"])
    qpos_min = checkpoint["qpos_min"]  # 形状 [28]
    qpos_max = checkpoint["qpos_max"]  # 形状 [28]
    mean=checkpoint["joints_mean"]
    std=checkpoint["joints_std"]
    model.eval()
    test_data = (retarget.joints.cpu() - mean) / std
    test_data = test_data.cuda()

    with torch.no_grad():
        pred = model(test_data)
        pred_qpos  = dataset.denormalize_qpos(pred.cpu())
    
    # 可视化
    print(pred_qpos.shape,retarget.joints.shape,retarget.full_qpos.shape)

    retarget.full_qpos[:,6:10] = pred_qpos 
    retarget.qpos = pred_qpos
    print(retarget.qpos.shape)
    retarget.render_mano_data()

if __name__ == "__main__":
    main()
