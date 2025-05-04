import h5py
import torch
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from utils.hand_model import ManoModel
from retarget_utils.GPU_optimizer import GPUPositionOptimizer 
from retarget_utils.robot_wrapper import RobotWrapper
import yaml
from pathlib import Path
from config.path_config import Path_config

class RetargetDatasetGenerator:
    def __init__(self, config):
        self.batch_size = 1024  # 每批生成数量
        self.total_samples = 100000  # 总样本数

        print(config)
        # 初始化MANO模型
        self.hand_model = ManoModel(
        mano_root='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano', 
        contact_indices_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano/contact_indices.json', 
        pose_distrib_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/Mano/mano/pose_distrib.pt', 
        device="cuda:0"
        )
        # 初始化优化器
        self.data_path = Path(Path_config.get_data_path())
        robot_file_path = str(self.data_path / "urdf/dex_retargeting/hands/shadow_hand/shadow_hand_right_glb_w_dummy.urdf")
        self.robot = RobotWrapper(robot_file_path)
        self.optimizer = GPUPositionOptimizer(
            self.robot,
            config["retargeting"]["target_joint_names"],
            config["retargeting"]["target_link_names"],
            config["retargeting"]["target_link_human_indices"],
            )
        # optimized_qpos = self.optimizer.batch_retarget(
        #     ref_pos=ref_pos,
        #     fixed_qpos=fixed_qpos,
        #     last_qpos=last_qpos
        # )

    def generate_dataset(self, output_path):
        self.robot.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 创建HDF5文件
        with h5py.File(output_path, 'w') as f:
            input_ds = f.create_dataset(
                "input_poses", 
                (self.total_samples, 48),
                dtype='float32'
            )
            output_ds = f.create_dataset(
                "output_angles",
                (self.total_samples, len(self.optimizer.idx_pin2target)),
                dtype='float32'
            )
            meta_ds = f.create_dataset(
                "metadata",
                (self.total_samples, 3),  # [loss, status, iter]
                dtype='float32'
            )

            for batch_idx in tqdm(range(0, self.total_samples, self.batch_size)):
                batch_poses = torch.randn(
                    self.batch_size, 48, 
                    device=self.hand_model.device
                ) * 0.5
                
                # 单进程处理
                results = []
                for pose in batch_poses.cpu().numpy():
                    results.append(self.process_sample(pose,robot=self.robot))
                
                end_idx = min(batch_idx + self.batch_size, self.total_samples)
                input_ds[batch_idx:end_idx] = batch_poses.cpu().numpy()
                output_ds[batch_idx:end_idx] = np.array([r[0] for r in results])
                meta_ds[batch_idx:end_idx] = np.array([r[1] for r in results])
            # # 分批次生成
            # for batch_idx in tqdm(range(0, self.total_samples, self.batch_size)):
            #     # 生成随机姿态
            #     batch_poses = torch.randn(
            #         self.batch_size, 48,
            #         device=self.hand_model.device
            #     ) * 0.5
                
            #     # 并行处理批数据
            #     with Pool(self.num_workers) as pool:
            #         results = pool.map(
            #             partial(self.process_sample, robot=self.robot),
            #             batch_poses.cpu().numpy()
            #         )

            #     # 存入HDF5
            #     end_idx = min(batch_idx + self.batch_size, self.total_samples)
            #     input_ds[batch_idx:end_idx] = batch_poses.cpu().numpy()
            #     output_ds[batch_idx:end_idx] = np.array([r[0] for r in results])
            #     meta_ds[batch_idx:end_idx] = np.array([r[1] for r in results])

    def process_sample(self,pose, robot):
        """单样本处理函数"""
        # 转换为完整姿态
        full_pose = torch.cat([
            torch.zeros(3),
            torch.tensor(pose),
            torch.zeros(3)
        ]).unsqueeze(0).to(robot.device)
        
        # 获取目标位置
        self.hand_model.set_parameters(full_pose, data_origin="ycb")
        verts = self.hand_model.get_trans_trimesh_data(i=0, pose=None)
        ref_pos = verts
        
        # 优化求解

        result = self.optimizer.batch_retarget(
            ref_pos,
            fixed_qpos=robot.default_fixed_joints,
            last_qpos=robot.last_qpos
        )
        return (result, [0, 1.0, 10])  # 成功样例

class ParallelOptimizerWrapper:
    """GPU加速的并行优化器"""
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer
        self.batch_optimizer = self._convert_to_batch()
        
    def _convert_to_batch(self):
        # 实现批量优化逻辑
        pass


def load_yaml(path: Path):
    """通用的 YAML 加载方法"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, "r") as file:
        return yaml.safe_load(file)
# 使用示例

if __name__ == "__main__":
    base_dir = Path(Path_config.BASE_DIR)
    config = load_yaml(base_dir / "config/mapping_config/shadow_hand_right.yml")

    
    generator = RetargetDatasetGenerator(config)
    generator.generate_dataset("retarget_dataset.h5")