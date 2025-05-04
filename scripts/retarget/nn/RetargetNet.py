import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class FingerLSTM(nn.Module):
    def __init__(self, input_dim=63, output_dim=4, lower=None, upper=None):
        super(FingerLSTM, self).__init__()
        self.lower_joint_limits = lower[6:10]
        self.upper_joint_limits = upper[6:10]

        # 增强特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            
            ResidualBlock(512),  # 新增残差块
            ResidualBlock(512),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        # 改进的LSTM结构
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=512,        # 增大隐层维度
            num_layers=3,           # 增加层深
            batch_first=True,
            bidirectional=True,     # 启用双向
            dropout=0.2,            # 添加层间dropout
            proj_size=128           # 使用投影降维
        )
        
        # 动态关节解码器
        self.joint_decoder = nn.Sequential(
            nn.Linear(128 * 2, 64),   # 双向特征拼接
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # 物理约束模块
        self.constraint_layer = PhysicalConstraintLayer(
            lower_limits=self.lower_joint_limits,
            upper_limits=self.upper_joint_limits
        )

        # 初始化策略
        self._init_weights()

    def _init_weights(self):
        # LSTM参数正交初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
                param.data.mul_(0.01)  # 缩小递归权重
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

        # 线性层Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)  # [B, 256]
        seq = x.unsqueeze(1).expand(-1, 4, -1)  # [B, 4, 256]
        lstm_out, _ = self.lstm(seq)  # [B, 4, 128]
        
        joints = self.joint_decoder(lstm_out).squeeze(-1)  # [B, 4]
        return self.constraint_layer(joints)

class ResidualBlock(nn.Module):
    """带门控机制的残差块"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, x):
        gate = self.gate(x)
        transformed = self.transform(x)
        return x * gate + transformed

class PhysicalConstraintLayer(nn.Module):
    """可学习的物理约束层"""
    def __init__(self, lower_limits, upper_limits):
        super().__init__()
        self.register_buffer('lower', lower_limits)
        self.register_buffer('upper', upper_limits)
        self.scale = nn.Parameter(torch.ones_like(lower_limits))
        self.bias = nn.Parameter(torch.zeros_like(lower_limits))
        
    def forward(self, x):
        # 带可学习参数的约束映射
        scaled = torch.sigmoid(x) * (self.upper - self.lower) + self.lower
        return self.scale * scaled + self.bias
    

class RetargetDataset(Dataset):
    def __init__(self, joints, qpos, train_ratio=0.9):
        # 数据分割
        split_idx = int(len(joints) * train_ratio)
        
        # 训练集处理
        self.train_joints = joints[:split_idx]
        self.train_qpos = qpos[:split_idx]
        
        # 验证集处理
        self.val_joints = joints[split_idx:]
        self.val_qpos = qpos[split_idx:]
        
        # 训练集标准化参数计算
        self.joints_mean = self.train_joints.view(-1, 3).mean(dim=0)
        self.joints_std = self.train_joints.view(-1, 3).std(dim=0)
        self.qpos_min = self.train_qpos.min(dim=0)[0]
        self.qpos_max = self.train_qpos.max(dim=0)[0]
        
        # 应用标准化
        self.train_joints = self.normalize_joints(self.train_joints)
        self.train_qpos = self.normalize_qpos(self.train_qpos)
        self.val_joints = self.normalize_joints(self.val_joints)
        self.val_qpos = self.normalize_qpos(self.val_qpos)
        
    def normalize_joints(self, data):
        return (data - self.joints_mean) / (self.joints_std + 1e-8)
    def denormalize_joints(self, data,std,mean):
        return data * std + mean

    def normalize_qpos(self, data):
        return (data - self.qpos_min) / (self.qpos_max - self.qpos_min + 1e-8)
    def denormalize_qpos(self, data):
        return data * (self.qpos_max[6:10] - self.qpos_min[6:10]) + self.qpos_min[6:10]
    
    def __len__(self):
        return len(self.train_joints)
    
    def __getitem__(self, idx):
        return self.train_joints[idx], self.train_qpos[idx]
    
    
    @property
    def validation_set(self):
        return self.val_joints, self.val_qpos
    
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class FingerNet(nn.Module):
    def __init__(self, input_dim=12, output_dim=4, lower=None, upper=None):
        super().__init__()
        self.lower = lower[9:13]  # 单指关节角度下限 [4]
        self.upper = upper[9:13] # 单指关节角度上限 [4]
        
        # 单指关节连接矩阵 (4个关节链式连接)
        self.adj = torch.tensor([
            [1,1,0,0],  # 关节0连接0和1
            [1,1,1,0],  # 关节1连接0,1,2
            [0,1,1,1],  # 关节2连接1,2,3
            [0,0,1,1]   # 关节3连接2,3
        ], dtype=torch.float32)
        
        # 编码器
        self.encoder = nn.Sequential(
            GraphConvBlock(3, 32, self.adj),
            nn.LayerNorm(32),
            nn.GELU(),
            GraphConvBlock(32, 64, self.adj),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 4)
        )
        
        # 物理约束
        self.constraint = SafeConstraintLayer(self.lower, self.upper)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=0.01)

    def forward(self, x):
        """
        输入: [B, 4, 3] (4个关节的xyz坐标)
        输出: [B, 4] (4个关节角度)
        """
        x = self.encoder(x)        # [B,4,64]
        x = x.mean(dim=1)          # [B,64]
        x = self.decoder(x)        # [B,4]
        return self.constraint(x)  # 应用约束

class GraphConvBlock(nn.Module):
    """适用于单指的图卷积块"""
    def __init__(self, in_dim, out_dim, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        agg = torch.matmul(self.adj, x)  # 邻域聚合 [B,4,in_dim]
        return self.linear(agg)          # [B,4,out_dim]

class SafeConstraintLayer(nn.Module):
    """单指关节约束"""
    def __init__(self, lower, upper, margin=0.05):
        super().__init__()
        self.register_buffer('scale', upper - lower - 2*margin)
        self.register_buffer('offset', lower + margin)
        
    def forward(self, x):
        return torch.sigmoid(x) * self.scale + self.offset