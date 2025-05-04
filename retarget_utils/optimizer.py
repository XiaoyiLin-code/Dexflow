from abc import abstractmethod
from typing import List, Optional
import nlopt
import numpy as np
import torch
from retarget_utils.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor
from retarget_utils.robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof
        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(f"Joint {target_joint_name} given does not appear to be in robot XML.")
            print(f"Target joint name: {target_joint_name}")
            print(f"Joint names: {joint_names}")
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        print(f"Target joint names: {self.target_joint_names}")
        print(idx_pin2target)
        self.idx_pin2target = np.array(idx_pin2target)
        self.idx_pin2fixed = np.array([i for i in range(robot.dof) if i not in idx_pin2target], dtype=int)
        self.opt = nlopt.opt(nlopt.GN_CRS2_LM, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)
        self.target_link_human_indices = target_link_human_indices
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6
        self.adaptor: Optional[KinematicAdaptor] = None
    
    def batch_setup(self, batch_size: int, joint_limits: np.ndarray, epsilon=1e-3):
        # 根据批量大小更新 opt_dof
        self.opt_dof *= batch_size
        # 重新初始化优化器
        self.opt = nlopt.opt(nlopt.GN_CRS2_LM, self.opt_dof)

        # 打印调试信息，确保维度正确
        #print((joint_limits[:, 1] + epsilon).repeat(batch_size).shape, self.opt_dof, joint_limits)

        # 对每个自由度的上下限进行重复处理，确保每个自由度的上下限正确处理
        lower_bounds = np.tile(joint_limits[:, 0] - epsilon, batch_size)
        upper_bounds = np.tile(joint_limits[:, 1] + epsilon, batch_size)

        # 打印 lower_bounds 和 upper_bounds 确保它们是正确的
        #print("Lower bounds:", lower_bounds)
        #print("Upper bounds:", upper_bounds)

        # 设置优化器的上下限
        self.opt.set_lower_bounds(lower_bounds.tolist())
        self.opt.set_upper_bounds(upper_bounds.tolist())




    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())
        self.lower_bounds = joint_limits[:, 0]
        self.upper_bounds = joint_limits[:, 1]

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        print(last_qpos)
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            print(f"Optimizer has {self.idx_pin2fixed} joints but non_target_qpos {fixed_qpos} is given")
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError:
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]

class TemporalContextManager:
    """滑动窗口时序上下文管理"""
    def __init__(self, window_size=3):
        self.window = []
        self.window_size = window_size
        
    def update(self, frame_data):
        if len(self.window) >= self.window_size:
            self.window.pop(0)
        self.window.append(frame_data.copy())
        
    def get_window(self):
        return np.stack(self.window, axis=0) if self.window else None

class MotionOptimizerCore:
    """运动学优化核心组件"""
    def __init__(self):
        self.kinematic_chain = []
        self.joint_weights = np.ones(20)  # 关节权重矩阵
        
    def configure_weights(self, config):
        """配置关节权重矩阵"""
        self.joint_weights = np.array([
            0.8 if 'thumb' in name else 
            0.6 if 'index' in name else 
            0.4 for name in config
        ])

class EnhancedOptimizer(Optimizer):
    """增强型优化器基类"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_manager = TemporalContextManager()
        self.motion_core = MotionOptimizerCore()
        self.velocity_threshold = 120  # rad/s²

class PositionOptimizer(EnhancedOptimizer):
    retargeting_type = "ENHANCED_POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        temporal_weight=1e-1,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta
        self.target_link_indices = self.get_link_indices(target_link_names)
        self.opt.set_ftol_abs(1e-8)
        
        # 新增组件
        self.temporal_weight = temporal_weight
        self.motion_core.configure_weights(target_joint_names)
        self.kalman = JointKalmanFilter(dim=len(target_joint_names))
        self.dual_quat_interp = DualQuaternionInterpolator()
        
        # Initialize velocity detector
        self.velocity_detector = AccelerationDetector(threshold=50.0)  # You can adjust the threshold if needed

    def detect_anomaly(self, current_vel):

        return np.any(np.abs(current_vel) > self.velocity_threshold)

    def get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot.dof)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        self.i=0
        window_data = self.temporal_manager.get_window()
        torch_window = None
        if window_data is not None:
            if window_data.shape[0]==3:
                torch_window = torch.as_tensor(window_data)


            

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            # 运动学前向传播
            qpos[self.idx_pin2target] = x
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]
            self.robot.compute_forward_kinematics(qpos)
            
            # 核心残差计算
            target_link_poses = [self.robot.get_link_pose(index) for index in self.target_link_indices]
            body_pos = np.stack([pose[:3, 3] for pose in target_link_poses], axis=0)
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()
            
            # 时空联合优化项
            huber_distance_pos = self.huber_loss(torch_body_pos, torch_target_pos)
            
            # 新增时空差分约束
            if torch_window is not None:
                # 获取上一帧的身体位置（从 torch_window 获取）
                previous_joint_pos = torch_window[-2:]
                predicted_joint_pos = previous_joint_pos
                last_qpos_tensor = torch.tensor(last_qpos, dtype=torch.float32)
                predicted_joint_pos[:,:6]=last_qpos_tensor[:6]
                # if isinstance(predicted_joint_pos, torch.Tensor):
                #     predicted_joint_pos = predicted_joint_pos.detach().cpu().numpy()
                # self.robot.compute_forward_kinematics(predicted_joint_pos)
                # predict_body_pos = np.stack([self.robot.get_link_pose(index)[:3, 3] for index in self.target_link_indices], axis=0)
                # torch_previous_body_pos = torch.as_tensor(predict_body_pos)
                # 计算当前帧与上一帧之间的位置差异
                tensor_x=torch.tensor(x)
                temporal_diff = torch.linalg.norm(tensor_x[6:]-2*predicted_joint_pos[1,6:]+predicted_joint_pos[0,6:])
                temporal_loss = torch.mean(temporal_diff) * self.temporal_weight
            else:
                temporal_loss = torch.tensor(0.0)
            
            # 新增双重四元数插值约束
            # dual_quat_loss = self.dual_quat_interp.calculate_loss(
            #     current_poses=target_link_poses,
            #     prev_poses=self.temporal_manager.window[-1] if self.temporal_manager.window else None
            # )
            
            # 联合损失函数
            combined_loss = (
                15.0 * huber_distance_pos+
                0.05 * temporal_loss 
                # 0.3 * dual_quat_loss +
                # 0.0001 * torch.norm(torch.as_tensor(x - last_qpos[self.idx_pin2target]))
            )
            self.i+=1
            # if self.i%1000==0:
            #      print("Combined loss:", combined_loss.item(), "Huber loss:", huber_distance_pos.item(), "Temporal loss:", temporal_loss.item(),self.i)
            result = combined_loss.cpu().detach().item()
            # 梯度计算
            if grad.size > 0:
                # 雅可比计算逻辑
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)
                jacobians = np.stack(jacobians, axis=0)
                
                # 计算雅可比梯度
                grad_pos_term = torch.autograd.grad(huber_distance_pos, torch_body_pos)[0]
                grad_pos = grad_pos_term.cpu().numpy()
                
                # 时空约束梯度项
                if torch_window is not None:
                    temporal_grad = torch.autograd.grad(temporal_loss, torch_body_pos)[0]
                    grad_temporal = np.matmul(temporal_grad.numpy()[:, None, :], jacobians).sum(0)
                else:
                    grad_temporal = np.zeros_like(grad_pos)

                grad_qpos_pos = np.matmul(grad_pos[:, None, :], jacobians).mean(1).sum(0)
                grad_qpos_diff = np.matmul(grad_temporal[:, None, :], jacobians).mean(1).sum(0)
                grad_qpos = (
                    20.0 * grad_qpos_pos +
                    0.05 * grad_qpos_diff 
                    # 0.0001 * (x - last_qpos[self.idx_pin2target])
                )
                grad[:] = grad_qpos * self.motion_core.joint_weights  # 应用关节权重

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        current_vel = self.velocity_detector.calculate(ref_value)
        if self.detect_anomaly(current_vel):
            self.opt.set_ftol_abs(1e-6)  # 收紧收敛条件
            self.temporal_weight *= 0.5   # 降低时序约束权重
        result = super().retarget(ref_value, fixed_qpos, last_qpos)
        
        # 后处理步骤
        # smoothed = self.kalman.update(result[self.idx_pin2target])
        # result[self.idx_pin2target] = smoothed
        
        # 更新时序上下文
        self.temporal_manager.update(result)
        return result

class DualQuaternionInterpolator:
    """双重四元数插值模块"""
    def calculate_loss(self, current_poses, prev_poses):
        # 计算当前帧和前一帧的双重四元数插值损失
        if prev_poses is None:
            # print("警告: 当前为第一帧，没有前一帧数据，跳过插值计算。")
            return 0.0  # 或者可以根据需要设置为一个默认损失
            
        # 假设current_poses和prev_poses是[旋转矩阵, 平移向量]列表
        # 使用四元数来表示旋转
        current_rot = [pose[:3, :3] for pose in current_poses]
        prev_rot = [pose[:3, :3] for pose in prev_poses]
        
        current_quaternions = [R.from_matrix(rot).as_quat() for rot in current_rot]
        prev_quaternions = [R.from_matrix(rot).as_quat() for rot in prev_rot]
        
        # 计算四元数插值损失
        loss_rot = 0
        for current, prev in zip(current_quaternions, prev_quaternions):
            # 四元数差异度量，可以使用四元数之间的夹角
            diff = R.from_quat(current).inv() * R.from_quat(prev)
            loss_rot += diff.magnitude()  # 使用四元数差的模来度量旋转的差异
        
        # 计算平移插值损失
        current_translations = [pose[:3, 3] for pose in current_poses]
        prev_translations = [pose[:3, 3] for pose in prev_poses]
        
        loss_trans = np.mean([np.linalg.norm(c - p) for c, p in zip(current_translations, prev_translations)])
        
        # 结合旋转和平移的损失
        loss = loss_rot + loss_trans
        return loss

class JointKalmanFilter:
    """关节状态卡尔曼滤波器"""
    def __init__(self, dim):
        self.dim = dim
        # 初始化状态估计
        self.x = np.zeros(dim)  # 关节角度
        self.P = np.eye(dim) * 1e-3  # 状态协方差
        self.Q = np.eye(dim) * 1e-3  # 过程噪声
        self.R = np.eye(dim) * 1e-2  # 测量噪声
        self.H = np.eye(dim)  # 观测矩阵
        self.F = np.eye(dim)  # 状态转移矩阵

    def update(self, joint_angles):
        """预测和校正"""
        # 预测步骤
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        
        # 校正步骤
        y = joint_angles - np.dot(self.H, self.x)  # 观测误差
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # 测量协方差
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # 卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态估计
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))  # 更新协方差
        
        return self.x  # 返回更新后的关节角度

class AccelerationDetector:
    """角加速度检测器"""
    def __init__(self, threshold=50.0):
        self.threshold = threshold  # 定义加速度阈值

    def calculate(self, positions, dt=0.1):
        """计算角加速度并判断是否突变"""
        # 计算关节位置的差分（速度）
        velocity = np.diff(positions, axis=0) / dt
        
        # 计算速度的差分（加速度）
        acceleration = np.diff(velocity, axis=0) / dt
        
        # 判断加速度是否超过阈值（突变检测）
        anomaly_detected = np.any(np.abs(acceleration) > self.threshold)
        
        return acceleration if not anomaly_detected else np.zeros_like(acceleration)  # 如果有突变，返回零加速度
