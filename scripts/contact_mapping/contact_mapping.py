import torch
import re
import numpy as np
from collections import deque
from kaolin.metrics.pointcloud import sided_distance
from scipy.spatial import cKDTree
def apply_hysteresis_to_distance_seq(distance_seq, dis_on, dis_off):
        """
        对一个长度为 T 的距离序列应用滞后策略，返回 T 帧的布尔接触状态序列
        """
        T = len(distance_seq)
        contact_flags = [False] * T
        prev_state = False
        for t in range(T):
            d = distance_seq[t]
            if d < dis_on:
                state = True
            elif d > dis_off:
                state = False
            else:
                state = prev_state
            contact_flags[t] = state
            prev_state = state
        return contact_flags
def get_finger_position(batch_idx, mask,robot_pc):
    points = robot_pc[batch_idx, mask, :3]
    return points.mean(dim=0).cpu().numpy() if points.shape[0] > 0 else None

class ContactMapping:
    def __init__(self,upper_bound=0.1,lower_bound=0.01,device='cuda'):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.middle_bound = (upper_bound + lower_bound) / 2
        self.device = device

        self.smoother=Contact_Smoother()

    def set_up(self,robot_pc,object_pc,mask_names,link_names):
        self.robot_pc = robot_pc
        self.object_pc = object_pc
        self.mask_names = mask_names
        self.link_names = link_names

        # self.all_link_indices=self.robot_pc[0,:, 3].long()
        self.all_link_indices=self.robot_pc[0,:, 3].long()
        self.all_link_names=[link_names[idx] for idx in self.all_link_indices.tolist()]
        
        self.finger_mapping = {
                0: 'ff',
                1: 'mf',
                2: 'rf',
                3: 'th',
                4: 'lf',
        }
        self.finger_pattern =  {
                'ff': re.compile(mask_names[0], re.IGNORECASE),
                'mf': re.compile(mask_names[1], re.IGNORECASE),
                'rf': re.compile(mask_names[2], re.IGNORECASE),
                'th': re.compile(mask_names[3], re.IGNORECASE),
                'lf': re.compile(mask_names[4], re.IGNORECASE),
            }

    
    def get_conatct_map_single_frame(self):
        mask_names=self.mask_names
        target_points_all = self.robot_pc[:, :3].to(self.device)
        distance_all, object_contact_point_indices_all = sided_distance(target_points_all.unsqueeze(0),self.object_pc.to(self.device).unsqueeze(0))
        ff_mask = torch.tensor([bool(re.search(mask_names[0], name, re.IGNORECASE)) for name in self.all_link_names], dtype=torch.bool, device=self.device)
        mf_mask = torch.tensor([bool(re.search(mask_names[1], name, re.IGNORECASE)) for name in self.all_link_names], dtype=torch.bool, device=self.device)
        rf_mask = torch.tensor([bool(re.search(mask_names[2], name, re.IGNORECASE)) for name in self.all_link_names], dtype=torch.bool, device=self.device)
        lf_mask = torch.tensor([bool(re.search(mask_names[4], name, re.IGNORECASE)) for name in self.all_link_names], dtype=torch.bool, device=self.device)
        th_mask = torch.tensor([bool(re.search(mask_names[3], name, re.IGNORECASE)) for name in self.all_link_names], dtype=torch.bool, device=self.device)
        target_points_ff =  self.robot_pc[ff_mask, :3].to(self.device)
        target_points_mf =  self.robot_pc[mf_mask, :3].to(self.device)
        target_points_rf =  self.robot_pc[rf_mask, :3].to(self.device)
        target_points_lf =  self.robot_pc[lf_mask, :3].to(self.device)
        target_points_th =  self.robot_pc[th_mask, :3].to(self.device)

        distance_ff, object_contact_point_indices_ff = sided_distance(target_points_ff.unsqueeze(0),self.object_pc.to(self.device).unsqueeze(0))
        distance_mf, object_contact_point_indices_mf = sided_distance(target_points_mf.unsqueeze(0),self.object_pc.to(self.device).unsqueeze(0))
        distance_rf, object_contact_point_indices_rf = sided_distance(target_points_rf.unsqueeze(0),self.object_pc.to(self.device).unsqueeze(0))
        distance_lf, object_contact_point_indices_lf = sided_distance(target_points_lf.unsqueeze(0),self.object_pc.to(self.device).unsqueeze(0))
        distance_th, object_contact_point_indices_th = sided_distance(target_points_th.unsqueeze(0),self.object_pc.to(self.device).unsqueeze(0))
        distance_mask_all=torch.tensor( distance_all < self.middle_bound).squeeze(0)  # 直接对整个张量进行逐元素比较)
        distance_mask_ff=[distance<self.middle_bound for distance in distance_ff]
        distance_mask_mf=[distance<self.middle_bound for distance in distance_mf]
        distance_mask_rf=[distance<self.middle_bound for distance in distance_rf]
        distance_mask_lf=[distance<self.middle_bound for distance in distance_lf]
        distance_mask_th=[distance<self.middle_bound for distance in distance_th]
        object_contact_point_indices_all=object_contact_point_indices_all.squeeze(0)
        object_contact_point_indices_ff=object_contact_point_indices_ff.squeeze(0)
        object_contact_point_indices_mf=object_contact_point_indices_mf.squeeze(0)
        object_contact_point_indices_rf=object_contact_point_indices_rf.squeeze(0)
        object_contact_point_indices_lf=object_contact_point_indices_lf.squeeze(0)
        object_contact_point_indices_th=object_contact_point_indices_th.squeeze(0)
        object_contact_point_indices_all_filtered = object_contact_point_indices_all[distance_mask_all]
        object_contact_point_indices_ff_filtered = object_contact_point_indices_ff[distance_mask_ff]
        object_contact_point_indices_mf_filtered = object_contact_point_indices_mf[distance_mask_mf]
        object_contact_point_indices_rf_filtered = object_contact_point_indices_rf[distance_mask_rf]
        object_contact_point_indices_lf_filtered = object_contact_point_indices_lf[distance_mask_lf]
        object_contact_point_indices_th_filtered = object_contact_point_indices_th[distance_mask_th]
        
        distance_all_reverse, robot_contact_point_indices_all = sided_distance(self.object_pc[object_contact_point_indices_all_filtered].to(self.device).unsqueeze(0), self.robot_pc[:,:3].unsqueeze(0))
        robot_contact_point_mask = torch.zeros( self.robot_pc.shape[0], dtype=torch.bool, device=self.device)
        robot_contact_point_mask[robot_contact_point_indices_all] = True
        ff_mask_reverse = ff_mask & robot_contact_point_mask
        mf_mask_reverse = mf_mask & robot_contact_point_mask
        rf_mask_reverse = rf_mask & robot_contact_point_mask
        lf_mask_reverse = lf_mask & robot_contact_point_mask
        th_mask_reverse = th_mask & robot_contact_point_mask
        robot_contact_point_indices_ff = torch.where(ff_mask_reverse)[0] 
        robot_contact_point_indices_mf = torch.where(mf_mask_reverse)[0]
        robot_contact_point_indices_rf = torch.where(rf_mask_reverse)[0]
        robot_contact_point_indices_lf = torch.where(lf_mask_reverse)[0]
        robot_contact_point_indices_th = torch.where(th_mask_reverse)[0]
        
        indices_dict={
            "all":object_contact_point_indices_all_filtered,
            "ff":object_contact_point_indices_ff_filtered,
            "mf":object_contact_point_indices_mf_filtered,
            "rf":object_contact_point_indices_rf_filtered,
            # "lf":object_contact_point_indices_lf_filtered,
            "th":object_contact_point_indices_th_filtered,
        }
        map_dict={
            "ff":ff_mask,
            "mf":mf_mask,
            "rf":rf_mask,
            # "lf":lf_mask,
            "th":th_mask,
        }
        robot_indices_dict={
            "all":robot_contact_point_indices_all,
            "ff":robot_contact_point_indices_ff,
            "mf":robot_contact_point_indices_mf,
            "rf":robot_contact_point_indices_rf,
            # "lf":robot_contact_point_indices_lf,
            "th":robot_contact_point_indices_th,
        }
        if mask_names[4]!="None":
            indices_dict["lf"]=object_contact_point_indices_lf_filtered
            map_dict["lf"]=lf_mask
            robot_indices_dict["lf"]=robot_contact_point_indices_lf
        return indices_dict, map_dict,robot_indices_dict
   

    def get_contact_map_sqpence(self, dis_on=1e-4, dis_off=2.5e-4):
        """
        修改后的 get_contact_map 函数：
        输入：
            - self.robot_pc: (B, L, d) tensor，其中 B 为帧数，L 与 all_link_names 长度一致
            - object_pc: (B, N, 3) tensor
            - all_link_names: 长度为 L 的字符串列表，描述 self.robot_pc 中各 link 的名称
            - dis_on: 开启接触的阈值
            - dis_off: 结束接触的阈值
        输出：
            - indices_dict: 各 finger 的 object contact point 索引（列表，每个元素对应一帧）
            - mask_dict: 各 finger 对应的固定 mask
            - robot_indices_dict: 各 finger 的 robot contact point 索引（列表，每个元素对应一帧）
        """
        device = self.device
        B = self.robot_pc.shape[0]
        object_pc= self.object_pc.to(device)
        all_link_names = self.all_link_names
        # --- 1. 生成各个 finger 的 mask（基于 all_link_names） ---
        mask_all = torch.tensor([bool(re.search(r"distal", name, re.IGNORECASE)) for name in all_link_names],
                                dtype=torch.bool, device=device)
        ff_mask = torch.tensor([bool(re.search(r"ffdistal", name, re.IGNORECASE)) for name in all_link_names],
                                dtype=torch.bool, device=device)
        mf_mask = torch.tensor([bool(re.search(r"mfdistal", name, re.IGNORECASE)) for name in all_link_names],
                                dtype=torch.bool, device=device)
        rf_mask = torch.tensor([bool(re.search(r"rfdistal", name, re.IGNORECASE)) for name in all_link_names],
                                dtype=torch.bool, device=device)
        lf_mask = torch.tensor([bool(re.search(r"lfdistal", name, re.IGNORECASE)) for name in all_link_names],
                                dtype=torch.bool, device=device)
        th_mask = torch.tensor([bool(re.search(r"thdistal", name, re.IGNORECASE)) for name in all_link_names],
                                dtype=torch.bool, device=device)


        target_points_all = self.robot_pc[:, mask_all, :3].to(device)
        target_points_ff  = self.robot_pc[:, ff_mask,  :3].to(device)
        target_points_mf  = self.robot_pc[:, mf_mask,  :3].to(device)
        target_points_rf  = self.robot_pc[:, rf_mask,  :3].to(device)
        target_points_lf  = self.robot_pc[:, lf_mask,  :3].to(device)
        target_points_th  = self.robot_pc[:, th_mask,  :3].to(device)

        distance_all, obj_idx_all = sided_distance(target_points_all, object_pc)
        distance_ff,  obj_idx_ff  = sided_distance(target_points_ff, object_pc)
        distance_mf,  obj_idx_mf  = sided_distance(target_points_mf, object_pc)
        distance_rf,  obj_idx_rf  = sided_distance(target_points_rf, object_pc)
        distance_lf,  obj_idx_lf  = sided_distance(target_points_lf, object_pc)
        distance_th,  obj_idx_th  = sided_distance(target_points_th, object_pc)

        # --- 4. 计算每帧、每个 finger 的代表性距离（例如取该帧所有 candidate 中的最小值） ---
        def compute_rep_distance(distance_tensor):
            rep = []
            for b in range(B):
                d_vec = distance_tensor[b]
                # 如果当前帧有候选点，则取最小距离，否则赋予一个大值
                if d_vec.numel() > 0:
                    rep.append(d_vec.min().item())
                else:
                    rep.append(dis_off + 1e-6)
            return rep

        rep_distance_all = compute_rep_distance(distance_all)
        rep_distance_ff  = compute_rep_distance(distance_ff)
        rep_distance_mf  = compute_rep_distance(distance_mf)
        rep_distance_rf  = compute_rep_distance(distance_rf)
        rep_distance_lf  = compute_rep_distance(distance_lf)
        rep_distance_th  = compute_rep_distance(distance_th)

        # --- 5. 对每个 finger 的代表性距离序列应用滞后策略，得到每帧的接触状态 ---
        contact_all = apply_hysteresis_to_distance_seq(rep_distance_all, dis_on, dis_off)
        contact_ff  = apply_hysteresis_to_distance_seq(rep_distance_ff,  dis_on, dis_off)
        contact_mf  = apply_hysteresis_to_distance_seq(rep_distance_mf,  dis_on, dis_off)
        contact_rf  = apply_hysteresis_to_distance_seq(rep_distance_rf,  dis_on, dis_off)
        contact_lf  = apply_hysteresis_to_distance_seq(rep_distance_lf,  dis_on, dis_off)
        contact_th  = apply_hysteresis_to_distance_seq(rep_distance_th,  dis_on, dis_off)
        contact_states = {
                'ff': contact_ff, 'mf': contact_mf,
                'rf': contact_rf, 'lf': contact_lf,
                'th': contact_th
            }

        # --- 6. 运动学平滑处理 ---
        finger_masks = {
            'ff': ff_mask, 'mf': mf_mask,
            'rf': rf_mask, 'lf': lf_mask,
            'th': th_mask
        }
        smoothed_contacts = self.smoother.apply_kinematic_smoothing(
            contact_states_dict=contact_states,
            robot_pc=self.robot_pc,
            finger_masks_dict=finger_masks,
        )

        # 更新接触状态
        contact_ff = smoothed_contacts['ff']
        contact_mf = smoothed_contacts['mf']
        contact_rf = smoothed_contacts['rf']
        contact_lf = smoothed_contacts['lf']
        contact_th = smoothed_contacts['th']

        indices_all_list, indices_ff_list = [], []
        indices_mf_list, indices_rf_list = [], []
        indices_lf_list, indices_th_list = [], []

        for b in range(B):
            if contact_all[b]:
                mask = distance_all[b] < dis_on
                indices_all_list.append(obj_idx_all[b][mask])
            else:
                indices_all_list.append(torch.tensor([], dtype=torch.long, device=device))
                
            if contact_ff[b]:
                mask = distance_ff[b] < dis_on
                indices_ff_list.append(obj_idx_ff[b][mask])
            else:
                indices_ff_list.append(torch.tensor([], dtype=torch.long, device=device))
                
            if contact_mf[b]:
                mask = distance_mf[b] < dis_on
                indices_mf_list.append(obj_idx_mf[b][mask])
            else:
                indices_mf_list.append(torch.tensor([], dtype=torch.long, device=device))
                
            if contact_rf[b]:
                mask = distance_rf[b] < dis_on
                indices_rf_list.append(obj_idx_rf[b][mask])
            else:
                indices_rf_list.append(torch.tensor([], dtype=torch.long, device=device))
                
            if contact_lf[b]:
                mask = distance_lf[b] < dis_on
                indices_lf_list.append(obj_idx_lf[b][mask])
            else:
                indices_lf_list.append(torch.tensor([], dtype=torch.long, device=device))
                
            if contact_th[b]:
                mask = distance_th[b] < dis_on
                indices_th_list.append(obj_idx_th[b][mask])
            else:
                indices_th_list.append(torch.tensor([], dtype=torch.long, device=device))

        indices_dict = [{
            "all": indices_all_list[i],
            "ff":  indices_ff_list[i],
            "mf":  indices_mf_list[i],
            "rf":  indices_rf_list[i],
            "lf":  indices_lf_list[i],
            "th":  indices_th_list[i],
        } for i in range(B)]

        mask_dict = [{
            "all": mask_all[i],
            "ff":  ff_mask[i],
            "mf":  mf_mask[i],
            "rf":  rf_mask[i],
            "lf":  lf_mask[i],
            "th":  th_mask[i],
        } for i in range(B)]

        # --- 7. 计算 self.robot_pc 上的 contact point 索引（对每帧单独处理） ---
        robot_idx_all_list = []
        robot_idx_ff_list  = []
        robot_idx_mf_list  = []
        robot_idx_rf_list  = []
        robot_idx_lf_list  = []
        robot_idx_th_list  = []
        for b in range(B):
            if contact_all[b]:
                # 根据 object contact 点在当前帧找对应的 robot contact 点
                obj_pts = object_pc[b][indices_all_list[b]].unsqueeze(0)    # (1, n_obj, 3)
                robot_pts = self.robot_pc[b, :, :3].unsqueeze(0)           # (1, L, 3)
                _, robot_idx_all = sided_distance(obj_pts, robot_pts)
                robot_idx_all = robot_idx_all.squeeze(0)
                # 构造 robot 的布尔 mask
                robot_mask = torch.zeros(self.robot_pc.shape[1], dtype=torch.bool, device=device)
                robot_mask[robot_idx_all] = True

                # 分别与各 finger 的 mask 求交集
                robot_idx_ff = torch.where(ff_mask & robot_mask)[0]
                robot_idx_mf = torch.where(mf_mask & robot_mask)[0]
                robot_idx_rf = torch.where(rf_mask & robot_mask)[0]
                robot_idx_lf = torch.where(lf_mask & robot_mask)[0]
                robot_idx_th = torch.where(th_mask & robot_mask)[0]

                robot_idx_all_list.append(robot_idx_all)
                robot_idx_ff_list.append(robot_idx_ff)
                robot_idx_mf_list.append(robot_idx_mf)
                robot_idx_rf_list.append(robot_idx_rf)
                robot_idx_lf_list.append(robot_idx_lf)
                robot_idx_th_list.append(robot_idx_th)
            else:
                # 当前帧无接触，则 robot 的各 finger 索引置为空
                robot_idx_all_list.append(torch.tensor([], dtype=torch.long, device=device))
                robot_idx_ff_list.append(torch.tensor([], dtype=torch.long, device=device))
                robot_idx_mf_list.append(torch.tensor([], dtype=torch.long, device=device))
                robot_idx_rf_list.append(torch.tensor([], dtype=torch.long, device=device))
                robot_idx_lf_list.append(torch.tensor([], dtype=torch.long, device=device))
                robot_idx_th_list.append(torch.tensor([], dtype=torch.long, device=device))

        robot_indices_dict_list = [{
            "all": robot_idx_all_list[i],
            "ff":  robot_idx_ff_list[i],
            "mf":  robot_idx_mf_list[i],
            "rf":  robot_idx_rf_list[i],
            "lf":  robot_idx_lf_list[i],
            "th":  robot_idx_th_list[i],
        } for i in range(B)]

        return indices_dict, mask_dict, robot_indices_dict_list
    
    def refine_robot_contact_point(self, robot_contact_point_indices, object_indices_dict, object_pc, object_mesh):
        """
        根据 object_indices_dict 和 object_pc 获取对应的接触点，找到 object_mesh 中距离接触点
        相近的点，计算这些点的法向均值，修正 robot_contact_point_indices 为法线最深的点，
        且与 robot_contact_point_indices 在同一个关节的 100 个点。
        """
        
        refined_indices = []

        # 将 object_pc 转换为 tensor 并移动到设备上
        object_pc_tensor = torch.tensor(object_pc, dtype=torch.float32, device=self.device)
        
        # 获取物体的所有顶点和法线（object_mesh）
        object_vertices = object_mesh.vertices  # shape: (num_vertices, 3)
        object_normals = object_mesh.vertex_normals  # shape: (num_vertices, 3)

        # 构建 KDTree 用于加速查找相近的点
        kdtree = cKDTree(object_vertices)
        
        for idx in robot_contact_point_indices:
            # 获取当前机器人接触点在 object_pc 中对应的点
            robot_contact_point = object_pc_tensor[idx]
            
            # 找到 object_mesh 中与 robot_contact_point 距离最小的点（我们使用 KNN 进行近邻查找）
            distances, nearest_indices = kdtree.query(robot_contact_point.cpu().numpy(), k=100)
            
            # 获取这 100 个点的法线
            nearest_normals = object_normals[nearest_indices]
            
            # 计算这些法线的均值
            mean_normal = nearest_normals.mean(axis=0)
            
            # 计算接触点相对于法线的方向（深度）
            projection = np.dot(mean_normal, (robot_contact_point.cpu().numpy() - object_vertices[nearest_indices].mean(axis=0)))
            
            # 根据法线方向选择法线最深的点（最远的点）
            # 这里假设 projection 是指向法线的方向，并且我们希望沿着法线深度方向选择最远的点
            # 计算法线反向的距离并选择最远的接触点
            depth_sorted_indices = np.argsort(distances)[::-1]
            sorted_nearest_indices = nearest_indices[depth_sorted_indices]
            
            # 修正机器人接触点索引为法线最深的 100 个点
            refined_indices.append(sorted_nearest_indices[:100])

        return refined_indices


from scipy.interpolate import CubicSpline    
class Contact_Smoother:
    def __init__(self,smoothing_window=5, speed_factor=0.5,tau_c=0.1,
                 beta1=0.5, v_max=0.1, delta_t=0.033):
        self.smoothing_window = smoothing_window
        self.speed_factor = speed_factor
        self.tau_c = tau_c       # 接触置信阈值
        self.beta1 = beta1      # 加速度敏感系数(需调参)
        self.v_max = v_max          # 最大允许速度(m/s)
        self.delta_t =  delta_t     # 时间间隔(30Hz)
        self.position_cache = {
            'ff': deque(maxlen=smoothing_window),
            'mf': deque(maxlen=smoothing_window),
            'rf': deque(maxlen=smoothing_window),
            'lf': deque(maxlen=smoothing_window),
            'th': deque(maxlen=smoothing_window),
        }

   
    def apply_kinematic_smoothing(
            self,
            contact_states_dict,  # 各指尖的接触状态字典 {'ff': list, 'mf': list,...}
            robot_pc,      # 机器人点云数据 (B, L, d)
            finger_masks_dict,    # 各指尖mask字典 {'ff': tensor, 'mf': tensor,...}
        ):
        B = robot_pc.shape[0]  # 总帧数
        fingers = ['ff', 'mf', 'rf', 'lf', 'th']
        smoothed_contacts = {f: np.array(contact_states_dict[f]) for f in fingers}

        # 辅助函数：获取指尖代表位置
        def get_finger_position(batch_idx, mask):
            points = robot_pc[batch_idx, mask, :3]
            return points.mean(dim=0).cpu().numpy() if points.shape[0] > 0 else None

        # 对每个指尖独立处理
        for finger in fingers:
            # 1. 收集历史位置数据
            position_cache = deque(maxlen=self.smoothing_window)
            for b in range(B):
                pos = get_finger_position(b, finger_masks_dict[finger])
                if pos is not None:
                    position_cache.append((b, pos))
                else:
                    position_cache.clear()  # 遇到无效帧清空缓存

            # 2. 滑动窗口处理
            valid_frames = [item[0] for item in position_cache]
            if len(valid_frames) < self.smoothing_window:
                continue  # 跳过数据不足的情况

            # 3. 三次样条拟合和状态修正
            for center_idx in range(2, len(valid_frames)-2):
                # 提取5帧窗口数据
                window = list(position_cache)[center_idx-2:center_idx+3]
                time_points = np.array([item[0]*self.delta_t for item in window])
                positions = np.array([item[1] for item in window])

                try:
                    # 各坐标轴独立拟合
                    cs = [CubicSpline(time_points, positions[:,i]) for i in range(3)]
                    
                    # 计算当前帧加速度（二阶导数）
                    current_time = window[2][0] * self.delta_t
                    accelerations = [cs_i(current_time, nu=2) for cs_i in cs]
                    acc_norm = np.linalg.norm(accelerations)
                    
                    # 接触可能性估计（公式7）
                    pc_t = 1 / (1 + np.exp(-self.beta1 * acc_norm))

                    # 速度估算（公式5）
                    prev_vel = (positions[2] - positions[1]) / self.delta_t
                    next_vel = (positions[3] - positions[2]) / self.delta_t
                    avg_vel = 0.5 * (prev_vel + next_vel)
                    vf_delta = np.linalg.norm(avg_vel) * self.delta_t

                    # 状态插补（公式4）
                    frame_idx = window[2][0]
                    prev_state = smoothed_contacts[finger][frame_idx-1]
                    next_state = smoothed_contacts[finger][frame_idx+1]
                    continuity_term = (prev_state + next_state)/2 + self.speed_factor*vf_delta

                    # 三阶段决策
                    if pc_t > 0.5 and np.linalg.norm(avg_vel) < self.v_max:
                        if continuity_term > self.tau_c:
                            smoothed_contacts[finger][frame_idx] = True
                        else:
                            # 保持原始滞后策略结果
                            pass
                    else:
                        smoothed_contacts[finger][frame_idx] = False
                except Exception as e:
                    print(f"Error processing {finger} frame {frame_idx}: {str(e)}")
                    continue

        return smoothed_contacts