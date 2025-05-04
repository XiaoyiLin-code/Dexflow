import torch
from typing import List, Optional

class PositionOptimizer:
    def __init__(self, 
                 robot,
                 target_joint_names: List[str], 
                 target_link_names: List[str], 
                 target_link_human_indices: List[int], 
                 device: str = 'cuda',
                 lr: float = 0.01, 
                 max_iter: int = 10000, 
                 tol: float = 1e-6,
                 huber_delta: float = 0.020,
                use_global_phase: bool = True,
                 global_tol: float = 1e-3,
                 global_iter: int = 200):
        self.use_global_phase = use_global_phase
        self.global_tol = global_tol
        self.global_iter = global_iter

        """
        robot: RobotWrapper
        target_joint_names: 目标关节名称
        target_link_names: 目标链接名称
        target_link_human_indices: 目标链接对应的人体索引
        device: 运行设备
        lr: 学习率
        max_iter: 最大迭代次数
        tol: 容差
        """
        self.robot = robot
        self.target_joint_names = target_joint_names
        self.target_link_names = target_link_names
        self.target_link_human_indices = target_link_human_indices
        self.device = device
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        
        joint_objects = self.robot.get_joints()  # 获取关节对象列表
        joint_names = [j.name for j in joint_objects]  # 提取名称
        print("关节名称:", joint_names,target_joint_names)
        # 验证目标关节是否存在
        self.target_joint_indices = [
            joint_names.index(joint_name)
            for joint_name in target_joint_names
        ]
        print("目标关节名称:", self.target_joint_names, self.target_joint_indices)

        def get_link_names(chain):
            names = []
            def _recurse(node):
                names.append(node.link.name)
                for child in node.children:
                    _recurse(child)
            _recurse(chain._root)
            return names
        
        self.link_names = get_link_names(self.robot)  # 保存为实例变量
        for link_name in target_link_names:
            if link_name not in self.link_names:
                raise ValueError(f"链接 '{link_name}' 不存在于URDF中")
        self.target_link_indices = [
            self.link_names.index(link_name)  
            for link_name in target_link_names
        ]
        lower_limits, upper_limits = self.robot.get_joint_limits()
        lower_limits = torch.tensor(lower_limits, device=self.device)  
        upper_limits = torch.tensor(upper_limits, device=self.device)  
        self.lower_limits = lower_limits[self.target_joint_indices]
        self.upper_limits = upper_limits[self.target_joint_indices]
        print("目标链接索引:", self.target_link_indices, self.target_link_names)

    def clamp_joints(self, joints: torch.Tensor) -> torch.Tensor:
        """
        限制关节角度范围
        joints: [batch_size, num_joints]
        返回: [batch_size, num_joints]
        """
        return torch.clamp(joints, self.lower_limits, self.upper_limits).to(self.device)


    def optimize(self, 
            target_pos: torch.Tensor, 
            initial_guess: Optional[torch.Tensor] = None,
            batch_size=1) -> torch.Tensor:
        """
        移除全局优化阶段的纯梯度优化版本，保持接口不变
        """
        # 初始化参数
        if initial_guess is None:
            joints = torch.zeros(batch_size, len(self.target_joint_indices), 
                            device=self.device, requires_grad=True).float()
        else:
            joints = torch.tensor(
            initial_guess.cpu().numpy(), 
            device=self.device,
            dtype=torch.float3,
            requires_grad=True
        )

        target_indices = torch.tensor(self.target_joint_indices, device=joints.device)
        self.target_indices = target_indices
        # 使用PyTorch的集合运算

        # 获取关节限制
        lower, upper = self.robot.get_joint_limits()
        lower = torch.tensor(lower, device=self.device).unsqueeze(0)
        upper = torch.tensor(upper, device=self.device).unsqueeze(0)

        # 配置优化器
        optimizer = torch.optim.AdamW(
            [joints], 
            lr=self.lr, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.5, 
            patience=50
        )

        best_loss = float('inf')
        best_joints = joints.clone()
        no_improve = 0

        # 优化循环
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            all_indices = torch.arange(28, device=joints.device)
            non_target_mask = ~torch.isin(all_indices, target_indices)
            non_target_indices = all_indices[non_target_mask]

            
            # 施加关节限制
            
            restored_joints = torch.zeros(
                (batch_size, 28),
                device=self.device
            )
            non_target_joints = restored_joints[:, non_target_indices].clone()  # 使用clone代替copy

            clamped_joints = self.clamp_joints(joints)
            restored_joints[:, target_indices] = joints
            restored_joints[:, non_target_indices] = non_target_joints

            poses = self.robot.forward_kinematics(restored_joints)
            positions = []
            for link_name in self.target_link_names:
                pos = poses[link_name].get_matrix()[:, :3, 3]
                positions.append(pos)
            positions = torch.stack(positions, dim=1)
            pos_loss = torch.norm(
                positions - target_pos[:, self.target_link_human_indices, :], 
                dim=-1
            ).mean()
            
            

            total_loss = pos_loss * 1.0 
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # 早停机制
            with torch.no_grad():
                if total_loss < best_loss - self.tol:
                    best_loss = total_loss
                    best_joints = joints.clone()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve >= 200 or total_loss < self.tol:
                    print(f"Early stop at {epoch}, loss: {best_loss:.6f}")
                    break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={total_loss.item():.4f}")
        print(best_joints.shape)
        best_joints=self.clamp_joints(best_joints)
        restored_joints[:, target_indices] = best_joints
        return restored_joints
    
import torch
import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Optional

class EnhancedPositionOptimizer(PositionOptimizer):
    def __init__(self, 
                 robot,
                 target_joint_names: List[str], 
                 target_link_names: List[str], 
                 target_link_human_indices: List[int], 
                 device: str = 'cuda',
                 lr: float = 0.01, 
                 max_iter: int = 10000, 
                 tol: float = 1e-6,
                 huber_delta: float = 0.020,
                 use_global_phase: bool = True,
                 global_tol: float = 1e-3,
                 global_iter: int = 2):
        super().__init__(
            robot, target_joint_names, target_link_names, 
            target_link_human_indices, device, lr, max_iter, tol, huber_delta
        )
        lower_limits, upper_limits = self.robot.get_joint_limits()
        lower_limits = torch.tensor(lower_limits, device=self.device)
        upper_limits = torch.tensor(upper_limits, device=self.device)
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.verbose = True

    def optimize(self, 
                target_pos: torch.Tensor, 
                initial_guess: Optional[torch.Tensor] = None,
                batch_size=1) -> torch.Tensor:
        """
        改进后的优化方法，整合全局优化阶段
        保持接口完全兼容
        """
        # 阶段1: 全局优化（当未提供初始猜测时）
        if initial_guess is None and self.use_global_phase:
            global_joints = self._global_optimization(target_pos)
            initial_guess = global_joints.unsqueeze(0).repeat(batch_size, 1)

        # 阶段2: 原始局部优化
        return super().optimize(target_pos, initial_guess, batch_size)

    def _global_optimization(self, target_pos: torch.Tensor) -> torch.Tensor:
            """带调试输出的全局优化阶段"""
            # 调试信息：初始化状态
            if self.verbose:
                print("\n=== Global Optimization Phase ===")
                print(f"Target Position: {target_pos.cpu().numpy()[0]}")
                print(f"Joint Limits: \nLower: {self.lower_limits.cpu().numpy()}"
                    f"\nUpper: {self.upper_limits.cpu().numpy()}")

            # 定义带调试的损失函数
            def loss_fn(x: np.ndarray) -> float:
                # 转换并限制关节角度
                joints = torch.tensor(x, device=self.device, dtype=torch.float32)
                clamped = self.clamp_joints(joints.unsqueeze(0))
                
                # 前向运动学
                poses = self.robot.forward_kinematics(clamped)
                
                # 计算位置损失
                positions = torch.stack([
                    poses[name].get_matrix()[:, :3, 3] 
                    for name in self.target_link_names
                ], dim=1)
                pos_loss = torch.norm(
                    positions - target_pos[:, self.target_link_human_indices, :], 
                    dim=-1
                ).mean()
                
                # 调试输出（每50次评估打印）
                if self.verbose and (loss_fn.eval_count % 50 == 0):
                    print(f"\nEvaluation #{loss_fn.eval_count}")
                    print(f"Position Loss: {pos_loss.item():.6f}")
                    
                loss_fn.eval_count += 1
                return pos_loss.item()
            loss_fn.eval_count = 0  # 初始化计数器

            # 定义回调函数
            def callback(xk, convergence):
                if self.verbose:
                    current_loss = loss_fn(xk)
                    joint_ranges = xk.ptp(axis=0)
                    print(f"\n--- Iteration {callback.iter_count} ---")
                    print(f"Best Loss: {current_loss:.6f}")
                 
                    callback.iter_count += 1
            callback.iter_count = 0

            # 执行优化
            bounds = list(zip(
                self.lower_limits.cpu().numpy(),
                self.upper_limits.cpu().numpy()
            ))
            result = differential_evolution(
                func=loss_fn,
                bounds=bounds,
                strategy='best1bin',
                maxiter=5,
                popsize=20,
                tol=self.global_tol,
                workers=1,
                callback=callback if self.verbose else None
            )

            # 最终调试输出
            if self.verbose:
                print("\n=== Global Optimization Result ===")
                print(f"Success: {result.success}")
                print(f"Message: {result.message}")
                print(f"Optimal Joints: {result.x.round(4)}")
                print(f"Final Loss: {result.fun:.6f}")
                print(f"Evaluations: {loss_fn.eval_count}")

            return torch.tensor(result.x, device=self.device)