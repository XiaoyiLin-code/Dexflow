import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.hand_model import create_hand_model
import torch

# 生成手模型
hand = create_hand_model("shadow", "right", torch.device("cpu"))

def compute_rotation_matrix1(from_vec1, from_vec2, from_vec3, to_vec1, to_vec2, to_vec3):
    # 单位化所有向量
    from_vec1 = from_vec1 / np.linalg.norm(from_vec1)
    from_vec2 = from_vec2 / np.linalg.norm(from_vec2)
    from_vec3 = from_vec3 / np.linalg.norm(from_vec3)
    to_vec1 = to_vec1 / np.linalg.norm(to_vec1)
    to_vec2 = to_vec2 / np.linalg.norm(to_vec2)
    to_vec3 = to_vec3 / np.linalg.norm(to_vec3)

    # 构造初始坐标系：x = from_vec1, y = from_vec2, z = from_vec3
    from_basis = np.stack((from_vec1, from_vec2, from_vec3), axis=0)

    # 构造目标坐标系：x = to_vec1, y = to_vec2, z = to_vec3
    to_basis = np.stack((to_vec1, to_vec2, to_vec3), axis=0)

    # 计算右乘旋转矩阵 R_{AB}，使得：from = R_{AB} @ to
    R_matrix = from_basis.T @ to_basis  # 注意这里是 from_basis.T @ to_basis
    print(to_basis, from_basis@R_matrix)

    # 将旋转矩阵转换为 Rotation 对象并返回
    rotation = R.from_matrix(R_matrix)
    r= rotation.as_euler('xyz', degrees=True)[0]
    p= rotation.as_euler('xyz', degrees=True)[1]
    y= rotation.as_euler('xyz', degrees=True)[2]
    Rx=np.array([[1, 0, 0],
                   [0, np.cos(np.radians(r)), -np.sin(np.radians(r))],
                   [0, np.sin(np.radians(r)), np.cos(np.radians(r))]])
    Ry=np.array([[np.cos(np.radians(p)), 0, np.sin(np.radians(p))],
                   [0, 1, 0],
                   [-np.sin(np.radians(p)), 0, np.cos(np.radians(p))]])
    Rz=np.array([[np.cos(np.radians(y)), -np.sin(np.radians(y)), 0],
                     [np.sin(np.radians(y)), np.cos(np.radians(y)), 0],
                     [0, 0, 1]])
    print(from_basis@Rz@Ry@Rx)
    print(from_vec1@Rz@Ry@Rx)
    print(from_vec2@Rz@Ry@Rx)
    print(from_vec3@Rz@Ry@Rx)
    return rotation


def compute_rotation_matrix(from_vec1, from_vec2, from_vec3, to_vec1, to_vec2, to_vec3):
    # 单位化所有向量
    from_vec1 = from_vec1 / np.linalg.norm(from_vec1)
    from_vec2 = from_vec2 / np.linalg.norm(from_vec2)
    from_vec3 = from_vec3 / np.linalg.norm(from_vec3)
    to_vec1 = to_vec1 / np.linalg.norm(to_vec1)
    to_vec2 = to_vec2 / np.linalg.norm(to_vec2)
    to_vec3 = to_vec3 / np.linalg.norm(to_vec3)

    # 构造初始坐标系：x = from_vec1, y = from_vec2, z = from_vec3
    from_basis = np.stack((from_vec1, from_vec2, from_vec3), axis=1)

    # 构造目标坐标系：x = to_vec1, y = to_vec2, z = to_vec3
    to_basis = np.stack((to_vec1, to_vec2, to_vec3), axis=1)

    # 旋转矩阵 R，使得：to = R @ from
    R_matrix = to_basis @ from_basis.T

    # 将旋转矩阵转换为 Rotation 对象并返回
    rotation = R.from_matrix(R_matrix)
    return rotation

def apply_local_rpy_rotations(from_vec1, from_vec2, from_vec3, roll, pitch, yaw):
    # 计算局部旋转（即逐步应用绕每个轴的旋转）
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.radians(roll)), -np.sin(np.radians(roll))],
                   [0, np.sin(np.radians(roll)), np.cos(np.radians(roll))]])
    Ry = np.array([[np.cos(np.radians(pitch)), 0, np.sin(np.radians(pitch))],
                   [0, 1, 0],
                   [-np.sin(np.radians(pitch)), 0, np.cos(np.radians(pitch))]])
    Rz = np.array([[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
                   [np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
                   [0, 0, 1]])

    # 应用局部旋转到每个基向量
    rotated_vec1 = from_vec1 @Rz@ Ry@Rx
    rotated_vec2 = from_vec2 @Rz@ Ry@Rx   
    rotated_vec3 = from_vec3 @Rz@ Ry@Rx   
    print("rotate",rotated_vec1, rotated_vec2, rotated_vec3)
    return rotated_vec1, rotated_vec2, rotated_vec3

def plot_vectors(from_vectors, to_vectors,  robot_pc=None, robot_pc2=None, label_from="Initial", label_to="Target",target=None):
    fig = plt.figure()
    print(from_vectors, to_vectors, target)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制初始坐标系的向量
    for vec in from_vectors:
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='r', length=1.0)

    # 绘制目标坐标系的向量
    for vec in to_vectors:
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='b', length=1.0)
    
    # 绘制目标坐标系的向量（绿色）
    if target is not None:
        for vec in target:
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='g', length=1.0)

    # 如果传入了机器人点云，则绘制它
    if robot_pc is not None:
        ax.scatter(robot_pc[:, 0], robot_pc[:, 1], robot_pc[:, 2], color='purple', label='Robot PC')

    if robot_pc2 is not None:
        ax.scatter(robot_pc2[:, 0], robot_pc2[:, 1], robot_pc2[:, 2], color='orange', label='Robot PC2')

    # 设置坐标轴范围
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend([label_from, label_to, "to_vectors", "Robot PC", "Robot PC2"])

    plt.show()

# 示例输入向量（初始向量与目标向量）
from_vec1 = np.array([1, 0, 0])  # 初始向量1 (X轴)
from_vec2 = np.array([0, 1, 0])  # 初始向量2 (Y轴)
from_vec3 = np.array([0, 0, 1])  # 初始向量3 (Z轴)

to_vec1 = np.array([ -5,-5,4])  # 目标向量1
to_vec1=to_vec1/np.linalg.norm(to_vec1)

to_vec2 = np.array([1,4,0])  # 目标向量2
to_vec2 = to_vec2 - np.dot(to_vec2, to_vec1) * to_vec1
to_vec2 = to_vec2 / np.linalg.norm(to_vec2)
to_vec3 = np.cross(to_vec1, to_vec2)  # 目标向量3（从目标向量1和目标向量2的叉积得到）

# 计算旋转矩阵并返回 Rotation 对象
rotation = compute_rotation_matrix1(from_vec1, from_vec2, from_vec3, to_vec1, to_vec2, to_vec3)

# 定义旋转角度
roll = rotation.as_euler('xyz', degrees=True)[0]
pitch = rotation.as_euler('xyz', degrees=True)[1]
yaw = rotation.as_euler('xyz', degrees=True)[2]

# 转换为弧度
roll_rad = -rotation.as_euler('xyz', degrees=False)[0]
pitch_rad = -rotation.as_euler('xyz', degrees=False)[1]  # 注意Y轴反向
yaw_rad = -rotation.as_euler('xyz', degrees=False)[2]  # 注意Z轴反向

print(f"roll: {roll}, pitch: {pitch}, yaw: {yaw}")

# 获取机器人点云（生成虚拟数据）
init_qpos = torch.zeros(28)
robot_pc = hand.get_transformed_links_gene_pc(
                torch.tensor(init_qpos, dtype=torch.float32).to(torch.device("cpu"))
        ).numpy()  # 转换为 NumPy 数组
init_qpos[3:6] = torch.tensor([roll_rad, pitch_rad, yaw_rad])  # 将局部旋转角度传入机器人模型
robot_pc2 = hand.get_transformed_links_gene_pc(
                torch.tensor(init_qpos, dtype=torch.float32).to(torch.device("cpu"))
        ).numpy()  # 转换为 NumPy 数组

# 随机采样100个点
robot_pc = robot_pc[np.random.choice(robot_pc.shape[0], 100, replace=False), :]
robot_pc2 = robot_pc2[np.random.choice(robot_pc2.shape[0], 100, replace=False), :]

# 可视化逐步旋转后的坐标系
rotated_vec1, rotated_vec2, rotated_vec3 = apply_local_rpy_rotations(from_vec1, from_vec2, from_vec3, roll, pitch, yaw)
plot_vectors([from_vec1, from_vec2, from_vec3], [rotated_vec1, rotated_vec2, rotated_vec3],  robot_pc, robot_pc2, label_from="Original", label_to="After RPY", target=[to_vec1, to_vec2, to_vec3])
