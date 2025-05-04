import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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







if __name__ == "__main__":
    # 配置路径
    data_dir = "/home/lightcone/workspace/Dexflow/scripts/retarget/vision_pro/traj_data/"
    file_name = "all_frames.npy"
    file_path = os.path.join(data_dir, file_name)
    
    # 加载数据
    motion_data = load_npy_data(file_path)
    
    if motion_data is not None:
        # 数据预处理
        motion_data = motion_data.astype(np.float32)
        
        # 坐标系调整（根据实际需要）
        motion_data[..., 1] += 0.5  # Y轴偏移示例
        
        # 可视化前100帧
        visualize_single_hand(motion_data[:], fps=120)