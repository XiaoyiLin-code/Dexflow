import torch
import pytorch_kinematics as pk

# 1. 加载URDF模型
urdf_path = "/home/lightcone/workspace/Dexflow/data/urdf/dex_retargeting/hands/shadow_hand/shadow_hand_right_glb_w_dummy.urdf"
with open(urdf_path, "r") as f:
    urdf_content = f.read()

# 替换为URDF中实际存在的末端链接名称（例如 "right_hand"）
chain = pk.build_chain_from_urdf(urdf_content)
print("关节数:", chain)
# 2. 生成批量输入
batch_size = 10000
joint_angles = torch.rand(batch_size, len(chain.get_joint_parameter_names()))

# 3. 计算批量FK
with torch.no_grad():
    transforms = chain.forward_kinematics(joint_angles)

# 打印所有键和对应的变换矩阵
for link_name, transform in transforms.items():
    # 获取变换矩阵张量（形状为 [batch_size, 4, 4]）
    transform_matrix = transform.get_matrix()
    print(f"链接名称: {link_name}")
    print(f"变换矩阵形状: {transform_matrix.shape}")
    print("-------------------")

