import torch
import numpy as np
from typing import List, Optional
from scipy.spatial.distance import cdist

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
                 pso_iter: int = 50,  # 粒子群优化的迭代次数
                 num_particles: int = 10  # 粒子数目
                 ):
        """
        robot: RobotWrapper
        target_joint_names: 目标关节名称
        target_link_names: 目标链接名称
        target_link_human_indices: 目标链接对应的人体索引
        device: 运行设备
        lr: 学习率
        max_iter: 最大迭代次数
        tol: 容差
        pso_iter: 粒子群优化的迭代次数
        num_particles: 粒子数目
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
        self.pso_iter = pso_iter
        self.num_particles = num_particles
        
        joint_objects = self.robot.get_joints()  # 获取关节对象列表
        joint_names = [j.name for j in joint_objects]  # 提取名称
        
        # 验证目标关节是否存在
        self.target_joint_indices = [
            joint_names.index(joint_name)
            for joint_name in target_joint_names
        ]
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
        print("目标链接索引:", self.target_link_indices, self.target_link_names)

    def clamp_joints(self, joints: torch.Tensor) -> torch.Tensor:
            """
            限制关节角度范围
            joints: [batch_size, num_joints]
            返回: [batch_size, num_joints]
            """
            lower_limits, upper_limits = self.robot.get_joint_limits()
            lower_limits = torch.tensor(lower_limits, device=self.device)
            upper_limits = torch.tensor(upper_limits, device=self.device)
            return torch.clamp(joints, lower_limits, upper_limits).to(self.device)



    def optimize(self, 
                target_pos: torch.Tensor, 
                initial_guess: Optional[torch.Tensor] = None,
                batch_size=1) -> torch.Tensor:
        """
        执行并行优化
        target_pos: [batch_size, num_links, 3]
        initial_guess: [batch_size, num_joints]
        返回: [batch_size, num_joints] 优化后的关节角度
        """
        # Step 1: 粒子群优化（全局搜索）
        def pso_optimizer_for_batch(batch_target_pos: torch.Tensor):
            batch_size = batch_target_pos.size(0)
            best_loss = torch.full((batch_size,), float('inf'), device=self.device)
            best_joints = torch.zeros_like(batch_target_pos)

            # Initialize particles: Shape: [batch_size, num_joints, num_particles]
            particles = torch.randn(batch_size, len(self.target_joint_indices), self.num_particles, device=self.device)
            velocities = torch.zeros_like(particles)
            
            # Initialize pbest positions and losses with correct dimensions
            pbest_positions = particles.clone()  # Shape: [batch_size, num_joints, num_particles]
            pbest_losses = torch.full((self.num_particles, batch_size), float('inf'), device=self.device)
            
            # Initialize gbest
            gbest_position = torch.zeros(batch_size, len(self.target_joint_indices), device=self.device)
            gbest_loss = torch.full((batch_size,), float('inf'), device=self.device)

            for _ in range(self.pso_iter):
                for i in range(self.num_particles):
                    # Current particle's positions: [batch_size, num_joints]
                    current_particle = particles[:, :, i]
                    loss = self._calculate_loss_for_batch(current_particle, batch_target_pos)
                    
                    # Update personal best
                    mask = loss < pbest_losses[i]
                    pbest_losses[i] = torch.where(mask, loss, pbest_losses[i])
                    # Corrected indexing for pbest_positions
                    pbest_positions[:, :, i] = torch.where(
                        mask.unsqueeze(1),  # Shape: [batch_size, 1]
                        current_particle,   # Shape: [batch_size, num_joints]
                        pbest_positions[:, :, i]  # Shape: [batch_size, num_joints]
                    )
                    
                    # Update global best
                    gbest_mask = loss < gbest_loss
                    gbest_loss = torch.where(gbest_mask, loss, gbest_loss)
                    gbest_position = torch.where(
                        gbest_mask.unsqueeze(1),  # Shape: [batch_size, 1]
                        current_particle,         # Shape: [batch_size, num_joints]
                        gbest_position            # Shape: [batch_size, num_joints]
                    )

                # Update velocities and positions
                w, c1, c2 = 0.5, 1.5, 1.5
                # Ensure pbest_positions and gbest_position are correctly shaped for broadcasting
                velocities = w * velocities + c1 * torch.rand_like(velocities) * (pbest_positions - particles) + c2 * torch.rand_like(velocities) * (gbest_position.unsqueeze(2) - particles)
                particles += velocities

            return gbest_position


        
        # Step 1: 对每个数据点执行粒子群优化并获取结果
        if initial_guess is None:
            initial_guess = pso_optimizer_for_batch(target_pos)
            initial_guess=initial_guess.unsqueeze(0)
            print(initial_guess.shape)
        # Step 2: 局部优化（Adam优化器）
        joints = initial_guess.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([joints], lr=self.lr)
        
        best_loss = float('inf')
        best_joints = joints.clone()

        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            
            # 施加关节限制
            clamped_joints = self.clamp_joints(joints)
            
            # 计算正向运动学
            poses = self.robot.forward_kinematics(clamped_joints)  # [batch, num_links, 4, 4]
            target_poses = [poses[link_name].get_matrix()[:, :3, 3] for link_name in self.target_link_names]
            positions = torch.stack(target_poses, dim=1)    # [batch, num_links, 3]

            # 计算位置损失
            loss = self.huber_loss(positions, target_pos[:, self.target_link_human_indices, :])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
            
            # 记录最佳解
            with torch.no_grad():
                if loss < best_loss - self.tol:
                    best_loss = loss
                    best_joints = joints.clone()
                elif abs(loss - best_loss) <= self.tol:
                    print("loss < tol")
                    break
        
        return self.clamp_joints(best_joints)

    def _calculate_loss_for_batch(self, joints: torch.Tensor, batch_target_pos: torch.Tensor) -> torch.Tensor:
        """
        计算目标位置与当前关节角度的损失（批量处理）
        """
        clamped_joints = self.clamp_joints(joints)
        poses = self.robot.forward_kinematics(clamped_joints)  # [batch, num_links, 4, 4]
        target_poses = [poses[link_name].get_matrix()[:, :3, 3] for link_name in self.target_link_names]
        positions = torch.stack(target_poses, dim=1)  # [batch, num_links, 3]
        
        return self.huber_loss(positions, batch_target_pos[:, self.target_link_human_indices, :])