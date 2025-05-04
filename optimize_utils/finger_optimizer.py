import time
import torch

class FingerOptimizer:
    def __init__(self,hand_name, hand_model=None, object_surface=None, contact_points_per_finger=None, initial_hand_pose=None,
                 finger_joints_indics=None, lr=0.01, max_steps=200, tolerance=1e-18, alpha=1.0, beta=0.1,
                 w_dis=0.001, w_pen=1e5, w_spen=10.0, w_joints=10):
        self.lr = lr
        self.hand_name = hand_name
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.alpha = alpha
        self.beta = beta
        self.w_dis = w_dis
        self.w_pen = w_pen
        self.w_spen = w_spen
        self.w_joints = w_joints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_up(self, hand_model, object_surface, contact_points_per_finger, initial_hand_pose, finger_joints_indices, object_contact_point_indices, object_pcs):
        self.hand_model = hand_model
        self.object_surface = object_surface
        self.contact_points_per_finger = contact_points_per_finger
        self.finger_joints_indices = finger_joints_indices
        self.initial_hand_pose = initial_hand_pose.clone().detach().float().to(self.device)  # 移动到 CUDA
        self.object_contact_point_indices = object_contact_point_indices
        self.object_pcs = object_pcs.squeeze(0).to(self.device)

    def forward_kinematics(self, contact_points, hand_pose):
        start_time = time.time()
        squared_distance, normals, distance_signs, closest_points, link_index, link_pcs = self.hand_model.compute_world_coordinates_and_normals(contact_points, hand_pose, self.object_surface)
        end_time = time.time()
        return squared_distance, normals, distance_signs, closest_points, link_index, link_pcs
    
    def get_finger_map(self):
        if self.hand_name=="shadow":
            return  {
            "palm": [0],
            "ff": [1, 2, 3], 
            "mf": [4, 5, 6],  
            "rf": [7, 8, 9], 
            "lf": [10, 11, 12, 13],  
            "th": [14, 15, 16],  
        }
        elif self.hand_name=="allegro":
            return {
            "ff": [1, 2, 3, 4, 5], 
            "mf": [6, 7, 8, 9, 10],  
            "rf": [11, 12, 13, 14, 15],  
            "th": [16, 17, 18, 19, 20],  
        }

    def compute_energies(self, contact_points, hand_pose, finger_joints, finger_name, joint_indices, last_hand_pose):
        finger_map = self.get_finger_map()
        if finger_name not in finger_map:
            raise ValueError(f"Invalid finger_name: {finger_name}. Must be one of {list(finger_map.keys())}")
        current_finger_indices = finger_map[finger_name]
        squared_distance, distance_signs, normals, closest_points, link_index ,link_pcs = self.forward_kinematics(contact_points, hand_pose)
        contact_points = contact_points.long()
        object_contact_point_indices_current = self.object_contact_point_indices[finger_name]

        # Calculate distance energy
        dist_start_time = time.time()
        robot_contact_candidates = link_pcs[contact_points.tolist()]
        object_contact_candidates = self.object_pcs[object_contact_point_indices_current]
        if len(object_contact_candidates.shape) == 3 and object_contact_candidates.shape[0] == 1:
            object_contact_candidates = object_contact_candidates.squeeze(0)
        E_dis = self.w_dis * torch.cdist(robot_contact_candidates, object_contact_candidates, p=2).mean()

        all_mask = torch.isin(link_index, torch.tensor(current_finger_indices, device=self.device))
        all_distance = squared_distance[all_mask]
        all_signs = distance_signs[all_mask]
        penetration_depth = all_distance[all_signs < 0]
        # if len(penetration_depth) != 0:
        #      print(f"penetration_depth: {max(penetration_depth)}")
        E_pen = self.w_pen * torch.sum(penetration_depth ** 2)
        if len(penetration_depth) != 0:
            E_pen = E_pen/len(penetration_depth)
        all_index_mask = torch.isin(link_index, torch.tensor(current_finger_indices, device=self.device))
        spen_keypoints = link_pcs
        current_spen_keypoints = spen_keypoints[all_index_mask].unsqueeze(0)
        other_spen_keypoints = spen_keypoints[~all_index_mask].unsqueeze(0)
        E_spen = self.w_spen * self.hand_model.self_penetration(current_spen_keypoints, other_spen_keypoints)
        E_regularization = 500 * self.w_joints * torch.sum(
            (hand_pose[joint_indices] - self.initial_hand_pose[joint_indices]) ** 2
        )
        E_dis = E_dis.to(self.device)
        E_pen = E_pen.to(self.device)
        E_spen = E_spen.to(self.device)
        E_regularization = E_regularization.to(self.device)

        # Adjust self penetration energy
        E_spen += 1

        # Total energy and loss
        # if E_pen>0.1:
        #     E_dis*=0
        total_energy = E_dis + E_pen + E_spen + E_regularization
        loss = E_dis + E_pen + E_spen
        # print(E_dis, E_pen, E_spen, E_regularization)
        #print(f"Finger: {finger_name}, E_dis: {E_dis.item():.4f}, E_pen: {E_pen.item():.4f}, E_spen: {E_spen.item():.4f}, E_regularization: {E_regularization.item():.4f}")
        end_time = time.time()
        #print(f"Total compute_energies execution time: {end_time - start_time:.6f} seconds")
        
        return total_energy, loss

    def optimize_finger(self, finger_name, initial_hand_pose, last_hand_pose=None):
        # return initial_hand_pose, 0
        start_time = time.time()
        contact_points = self.contact_points_per_finger[finger_name]
        # print(self.contact_points_per_finger)
        if contact_points is None or contact_points.numel() == 0:
            return initial_hand_pose, 0

        if contact_points.ndimension() == 0:
            if contact_points.item() is None:
                return initial_hand_pose, 0

        if contact_points.ndimension() > 0 and contact_points[0] is None:
            return initial_hand_pose, 0

        contact_points = torch.tensor(contact_points, dtype=torch.float32, device=self.device).requires_grad_(True)
        finger_joints = self.finger_joints_indices[finger_name]
        hand_pose = initial_hand_pose.clone()
        mask = torch.zeros_like(hand_pose, dtype=torch.bool)
        mask[finger_joints] = True
        hand_pose.requires_grad_(True)

        optimizer = torch.optim.Adam([hand_pose], lr=self.lr)
        prev_loss = float('inf')
        epsilon = 1e-8

        for step in range(self.max_steps):
            optimizer.zero_grad()
            masked_pose = hand_pose.clone()
            masked_pose[~mask] = initial_hand_pose[~mask]
            E_dis, finger_score = self.compute_energies(contact_points, masked_pose, finger_joints, finger_name, finger_joints, last_hand_pose)
            loss = E_dis * 100
            finger_score *= 100
            loss.backward()
            hand_pose.grad[~mask] = 0
            optimizer.step()
            relative_error = abs(loss.item() - prev_loss) / max(abs(loss.item()), epsilon)
            prev_loss = loss.item()

            if relative_error < self.tolerance:
                print(f"Converged after {step} steps")
                break

        end_time = time.time()
        print(f"optimize_finger execution time for {finger_name}: {end_time - start_time:.6f} seconds")
        return hand_pose.detach(), finger_score

    def optimize_all_fingers(self, last_hand_pose=None):
        optimized_hand_pose = self.initial_hand_pose.clone().detach()
        if last_hand_pose is not None:
            last_hand_pose = last_hand_pose.to(self.device)
        finger_scores = []
        for finger_name in self.contact_points_per_finger.keys():
            if finger_name == "all":
                continue
            optimized_hand_pose, finger_score = self.optimize_finger(finger_name, optimized_hand_pose, last_hand_pose)
            finger_scores.append(finger_score)
        print(optimized_hand_pose-self.initial_hand_pose)
        return optimized_hand_pose, finger_scores
