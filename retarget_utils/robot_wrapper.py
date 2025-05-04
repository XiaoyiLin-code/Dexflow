from typing import List
import trimesh
import numpy as np
import numpy.typing as npt
import pinocchio as pin


class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        #如果是urdf文件，就用pin.buildModelFromUrdf(path)来创建模型
        if path.endswith('.urdf'):
            print(path)
            models = pin.buildModelsFromUrdf(path)
            self.model: pin.Model = models[0]  # 获取第一个返回的 model
        elif path.endswith('.xml'):
            models = pin.buildModelsFromMJCF(path)
            self.model: pin.Model = models[0]  # 获取第一个返回的 model
        self.data: pin.Data = self.model.createData()
        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
            # Print the joint names and their corresponding nq values
        for i, joint in enumerate(self.model.joints):
            joint_name = self.model.names[i]  # Get the name of the joint
            print(f"Joint index: {i}, Name: {joint_name}, NQ: {joint.nq}")

        print(f"Joint name: {list(self.model.names)},  {[name for i, name in enumerate(self.model.names) if self.model.nqs[i] > 0]}")

        print(f"Robot {path} has {self.model.nq} joints and {self.model.nv} dofs.")
        if self.model.nv != self.model.nq:
            raise NotImplementedError(f"Can not handle robot with special joint.")


    # -------------------------------------------------------------------------- # 
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        # for name in self.link_names:
        #     try:
        #         frame_id = self.model.getFrameId(name, pin.BODY)
        #         print(f"[DEBUG] Frame name: {name}, Frame ID: {frame_id}")
        #     except Exception as e:
        #         print(f"[ERROR] Unable to get Frame ID for {name}: {e}")
        print(name)
        if name not in self.link_names:
            raise ValueError(f"{name} is not a link name. Valid link names: \n{self.link_names}")
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J
    
    def get_transformed_mesh(self, qpos: npt.NDArray) -> trimesh.Trimesh:
        """
        Compute forward kinematics and return the updated mesh with transformations.
        
        Args:
            qpos (np.ndarray): Joint positions
        
        Returns:
            trimesh.Trimesh: Combined mesh after applying transformations
        """
        self.compute_forward_kinematics(qpos)

        meshes = []
        for geom in self.geometry_model.geometryObjects:
            mesh_path = geom.meshPath
            try:
                # Load the mesh
                mesh = trimesh.load_mesh(mesh_path)
                # Get the frame transform of the link
                frame_id = geom.parentFrame
                transform = self.data.oMf[frame_id].homogeneous
                # Apply the transform to the mesh
                mesh.apply_transform(transform)
                meshes.append(mesh)
            except Exception as e:
                print(f"Failed to load or transform mesh {mesh_path}: {e}")

        if not meshes:
            raise ValueError("No meshes found or all meshes failed to load.")

        # Combine all transformed meshes into one
        combined_mesh = trimesh.util.concatenate(meshes)
        return combined_mesh

    
