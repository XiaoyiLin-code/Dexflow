import sys
import os
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import yaml
import open3d as o3d
import matplotlib.pyplot as plt
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.object_model import ObjectModel
from utils.hand_model import create_hand_model
from transforms3d.euler import euler2mat
from pytransform3d import rotations

def plane2pose(plane_parameters):
    r3 = plane_parameters[:3]
    r2 = torch.zeros_like(r3)
    if r3[2].item() ** 2 <= 0.5:
        r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0)
    else:
        r2[0], r2[1], r2[2] = (-r3[2], 0, r3[0])
    r1 = torch.cross(r2, r3)
    pose = torch.zeros([4, 4], dtype=torch.float, device=plane_parameters.device)
    pose[0, :3] = r1
    pose[1, :3] = r2
    pose[2, :3] = r3
    pose[2, 3] = plane_parameters[3]
    pose[3, 3] = 1
    return pose

class DexYCB_Dataset(Dataset):
    """
    PyTorch Dataset class for DexYCB dataset, aligned with DexYCBVideoDataset's object handling.
    """
    _SUBJECTS = [
        "20200709-subject-01",
        "20200813-subject-02",
        "20200820-subject-03",
        "20200903-subject-04",
        "20200908-subject-05",
        "20200918-subject-06",
        "20200928-subject-07",
        "20201002-subject-08",
        "20201015-subject-09",
        "20201022-subject-10",
    ]

    YCB_CLASSES = {
        1: "002_master_chef_can",
        2: "003_cracker_box",
        3: "004_sugar_box",
        4: "005_tomato_soup_can",
        5: "006_mustard_bottle",
        6: "007_tuna_fish_can",
        7: "008_pudding_box",
        8: "009_gelatin_box",
        9: "010_potted_meat_can",
        10: "011_banana",
        11: "019_pitcher_base",
        12: "021_bleach_cleanser",
        13: "024_bowl",
        14: "025_mug",
        15: "035_power_drill",
        16: "036_wood_block",
        17: "037_scissors",
        18: "040_large_marker",
        19: "051_large_clamp",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }

    def __init__(
        self,
        batch_size: int,
        data_dir: str,
        hand_type: str = "right",
        filter_objects: list = [],
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 5000,
        object_pc_type: str = 'random',
        noise_level: str = 'pointcloud',
        mano_root: str = None
    ):
        """
        Initializes the DexYCB_Dataset.

        Args:
            batch_size (int): Number of samples per batch.
            data_dir (str): Path to the DexYCB dataset directory.
            hand_type (str, optional): Type of hand ('right' or 'left'). Defaults to "right".
            filter_objects (list, optional): List of object names to filter. Defaults to [].
            is_train (bool, optional): Whether the dataset is for training. Defaults to True.
            debug_object_names (list, optional): List of object names for debugging. Defaults to None.
            num_points (int, optional): Number of points to sample from object point clouds. Defaults to 5000.
            object_pc_type (str, optional): Type of object point cloud sampling. Defaults to 'random'.
            noise_level (str, optional): Level of noise to add. Defaults to 'pointcloud'.
            mano_root (str, optional): Path to MANO model root directory. Defaults to None.
        """
        self.noise_level = noise_level
        self.batch_size = batch_size
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.debug_object_names = debug_object_names
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        device = 'cpu'

        self.hand_name = "mano"
        self.hand = create_hand_model(self.hand_name, torch.device('cpu'))

        self._data_dir = Path(data_dir)
        self._mano_dir = Path(mano_root)
        self._calib_dir = self._mano_dir / "calibration"
        self._model_dir = self._mano_dir / "models"
        self._graspdata_dir = self._data_dir / "captures"  # Adjust according to your directory structure

        # Initialize YCB_CLASSES as per DexYCBVideoDataset
        self.YCB_CLASSES = {
            1: "002_master_chef_can",
            2: "003_cracker_box",
            3: "004_sugar_box",
            4: "005_tomato_soup_can",
            5: "006_mustard_bottle",
            6: "007_tuna_fish_can",
            7: "008_pudding_box",
            8: "009_gelatin_box",
            9: "010_potted_meat_can",
            10: "011_banana",
            11: "019_pitcher_base",
            12: "021_bleach_cleanser",
            13: "024_bowl",
            14: "025_mug",
            15: "035_power_drill",
            16: "036_wood_block",
            17: "037_scissors",
            18: "040_large_marker",
            19: "051_large_clamp",
            20: "052_extra_large_clamp",
            21: "061_foam_brick",
        }

        # Load camera and MANO parameters
        self._intrinsics, self._extrinsics = self._load_camera_parameters()
        self._mano_side = hand_type
        self._mano_parameters = self._load_mano()

        # Load capture data similar to DexYCBVideoDataset
        self._load_capture_data(filter_objects)
        
        # Initialize object point clouds if needed
        self.object_pcs = {}
    
        # if self.object_pc_type != 'fixed':
        #     self.object_model = ObjectModel(
        #         data_root_path='/home/lightcone/workspace/DRO-retarget/Noise-learn/data/meshdata',
        #         batch_size_each=1,
        #         num_samples=0, 
        #         device=device
        #     )
        #     # Initialize object_model with unique object names
        #     self.object_model.initialize(self.object_names)

        #     # Collect scales from metadata
        #     scales = [
        #         [
        #             data['scale']
        #             for data in self.metadata[self.hand_name][object_name].values()
        #         ]
        #         for object_name in self.object_names
        #     ]
        #     max_len = max(len(sublist) for sublist in scales)

        #     # Pad scales to have the same length
        #     padded_scales = [
        #         sublist + [0.0] * (max_len - len(sublist))
        #         for sublist in scales
        #     ]

        #     # Convert to PyTorch tensor
        #     object_scale = torch.tensor(
        #         padded_scales, 
        #         dtype=torch.float, 
        #         device=device
        #     )
        #     self.object_model.object_scale_tensor = object_scale

        #     # Sample point clouds from object meshes
        #     for i, object_name in enumerate(self.object_names):
        #         mesh = self.object_model.get_mesh(i)
        #         object_pc, _ = mesh.sample(self.num_points, return_index=True)
        #         self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
        # else:
        #     print("!!! Using fixed object pcs !!!")

    def _load_camera_parameters(self):
        extrinsics = {}
        intrinsics = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("extrinsics"):
                continue
            extrinsic_file = cali_dir / "extrinsics.yml"
            name = cali_dir.stem[len("extrinsics_"):]
            with extrinsic_file.open(mode="r") as f:
                extrinsic = yaml.load(f, Loader=yaml.FullLoader)
            extrinsics[name] = extrinsic

        intrinsic_dir = self._calib_dir / "intrinsics"
        for intrinsic_file in intrinsic_dir.iterdir():
            with intrinsic_file.open(mode="r") as f:
                intrinsic = yaml.load(f, Loader=yaml.FullLoader)
            name = intrinsic_file.stem.split("_")[0]
            x = intrinsic["color"]
            camera_mat = np.array([[x["fx"], 0.0, x["ppx"]], 
                                   [0.0, x["fy"], x["ppy"]], 
                                   [0.0, 0.0, 1.0]])
            intrinsics[name] = camera_mat

        return intrinsics, extrinsics

    def _load_mano(self):
        mano_parameters = {}
        for cali_dir in self._calib_dir.iterdir():
            if not cali_dir.stem.startswith("mano"):
                continue

            mano_file = cali_dir / "mano.yml"
            with mano_file.open(mode="r") as f:
                shape_parameters = yaml.load(f, Loader=yaml.FullLoader)
            mano_name = "_".join(cali_dir.stem.split("_")[1:])
            mano_parameters[mano_name] = np.array(shape_parameters["betas"])

        return mano_parameters

    def _load_capture_data(self, filter_objects):
        """
        Loads capture data, filters based on the specified objects, and stores metadata and poses.

        Args:
            filter_objects (list): List of object names to filter.
        """
        # Filter setup
        self.use_filter = len(filter_objects) > 0
        inverse_ycb_class = {"_".join(value.split("_")[1:]): key for key, value in self.YCB_CLASSES.items()}
        ycb_object_names = list(inverse_ycb_class.keys())
        filter_ids = []
        for obj in filter_objects:
            if obj not in ycb_object_names:
                print(f"Filter object name {obj} is not a valid YCB name")
            else:
                filter_ids.append(inverse_ycb_class[obj])

        # Load subjects
        self._subject_dirs = [sub for sub in self._data_dir.iterdir() if sub.stem in self._SUBJECTS]
        self._capture_meta = {}
        self._capture_pose = {}
        self._capture_filter = {}
        self._captures = []
        self.object_names = set()

        for subject_dir in self._subject_dirs:
            for capture_dir in subject_dir.iterdir():
                meta_file = capture_dir / "meta.yml"
                if not meta_file.exists():
                    print(f"Meta file not found: {meta_file}")
                    continue
                with meta_file.open(mode="r") as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)

                if self.hand_name not in meta.get("mano_sides", []):
                    continue

                pose = np.load(capture_dir / "pose.npz", allow_pickle=True)
                if self.use_filter:
                    ycb_ids = meta.get("ycb_ids", [])
                    # Skip current capture if no desired object inside
                    if len(set(ycb_ids) & set(filter_ids)) < 1:
                        continue
                    capture_filter = [i for i in range(len(ycb_ids)) if ycb_ids[i] in filter_ids]
                    object_pose = pose["pose_y"]
                    frame_indices, filter_id = self._filter_object_motion_frame(capture_filter, object_pose)
                    if len(frame_indices) < 20:
                        continue
                    self._capture_filter[capture_dir.stem] = [filter_id]
                self._capture_meta[capture_dir.stem] = meta
                self._capture_pose[capture_dir.stem] = pose
                self._captures.append(capture_dir.stem)

                # Collect object names
                for ycb_id in meta.get("ycb_ids", []):
                    self.object_names.add(self.YCB_CLASSES.get(ycb_id, "unknown"))

        self.object_names = list(self.object_names)

    def _filter_object_motion_frame(self, capture_filter, object_pose, frame_margin=40):
        """
        Filters frames where the object is moving, with a specified margin.

        Args:
            capture_filter (list): Indices of objects to filter.
            object_pose (np.ndarray): Pose data for objects.
            frame_margin (int, optional): Number of frames to include before and after motion. Defaults to 40.

        Returns:
            tuple: (frames indices array, filter_id)
        """
        frames = np.arange(0)
        for filter_id in capture_filter:
            filter_object_pose = object_pose[:, filter_id, :]
            object_move_list = []
            for frame in range(filter_object_pose.shape[0] - 2):
                object_move_list.append(self.is_object_move(filter_object_pose[frame:, :]))
            if True not in object_move_list:
                continue
            first_frame = object_move_list.index(True)
            last_frame = len(object_move_list) - object_move_list[::-1].index(True) - 1
            start = max(0, first_frame - frame_margin)
            end = min(filter_object_pose.shape[0], last_frame + frame_margin)
            frames = np.arange(start, end)
            break
        return frames, filter_id

    @staticmethod
    def is_object_move(single_object_pose: np.ndarray):
        """
        Determines if the object is moving based on its translation.

        Args:
            single_object_pose (np.ndarray): Pose data for a single object.

        Returns:
            bool: True if the object is moving, False otherwise.
        """
        single_object_trans = single_object_pose[:, 4:]
        if single_object_trans.shape[0] < 2:
            return False
        future_frame = min(single_object_trans.shape[0] - 1, 5)
        current_move = np.linalg.norm(single_object_trans[1] - single_object_trans[0]) > 2e-2
        future_move = np.linalg.norm(single_object_trans[future_frame] - single_object_trans[0]) > 5e-2
        return current_move or future_move

    def _object_mesh_file(self, object_id):
        """
        Retrieves the file path for the object's mesh.

        Args:
            object_id (int): YCB object ID.

        Returns:
            str: Path to the object's mesh file.
        """
        obj_file = self._model_dir / self.YCB_CLASSES.get(object_id, "unknown") / "textured_simple.obj"
        if not obj_file.exists():
            raise FileNotFoundError(f"Object mesh file not found: {obj_file}")
        return str(obj_file.resolve())

    def __len__(self):
        return len(self._captures)

    def __getitem__(self, item):
        if item >= self.__len__():
            raise ValueError(f"Index {item} out of range")

        capture_name = self._captures[item]
        meta = self._capture_meta[capture_name]
        pose = self._capture_pose[capture_name]
        hand_pose = pose["pose_m"]
        object_pose = pose["pose_y"]
        ycb_ids = meta.get("ycb_ids", [])

        # Load extrinsic and MANO parameters
        extrinsic_name = meta.get("extrinsics", "")
        if extrinsic_name not in self._extrinsics:
            raise KeyError(f"Extrinsic '{extrinsic_name}' not found in loaded extrinsics.")
        extrinsic_mat = np.array(self._extrinsics[extrinsic_name]["extrinsics"]["apriltag"]).reshape([3, 4])
        extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)
        mano_name = meta.get("mano_calib", [""])[0]
        if mano_name not in self._mano_parameters:
            raise KeyError(f"MANO calibration '{mano_name}' not found in loaded MANO parameters.")
        mano_parameters = self._mano_parameters[mano_name]

        if self.use_filter:
            capture_filter = np.array(self._capture_filter.get(capture_name, []))
            if len(capture_filter) == 0:
                raise ValueError(f"No capture_filter found for capture '{capture_name}'")
            frame_indices, _ = self._filter_object_motion_frame(capture_filter, object_pose)
            ycb_ids = [ycb_ids[valid_id] for valid_id in self._capture_filter[capture_name]]
            hand_pose = hand_pose[frame_indices]
            object_pose = object_pose[frame_indices][:, capture_filter, :]
        
        object_mesh_files = [self._object_mesh_file(ycb_id) for ycb_id in ycb_ids]

        # Sample point clouds if required
        object_pcs = []
        for obj_name in [self.YCB_CLASSES.get(ycb_id, "unknown") for ycb_id in ycb_ids]:
            if self.object_pc_type == 'random':
                indices = torch.randperm(self.num_points)[:self.num_points]
                object_pc = self.object_pcs[obj_name][indices]
                object_pcs.append(object_pc)
            elif self.object_pc_type == 'fixed':
                object_pc = self.object_pcs[obj_name]
                object_pcs.append(object_pc)
            else:
                raise ValueError(f"Unknown object_pc_type: {self.object_pc_type}")

        ycb_data = dict(
            hand_pose=hand_pose,
            object_pose=object_pose,
            extrinsics=extrinsic_mat,
            ycb_ids=ycb_ids,
            hand_shape=mano_parameters,
            object_mesh_file=object_mesh_files,
            capture_name=capture_name,
            object_pcs=object_pcs  # Added object point clouds
        )
        return ycb_data

    def get_object_point_clouds(self, ycb_ids):
        """
        Retrieves point clouds for the given YCB IDs.

        Args:
            ycb_ids (list): List of YCB object IDs.

        Returns:
            list: List of point clouds corresponding to the YCB IDs.
        """
        object_pcs = []
        for ycb_id in ycb_ids:
            obj_name = self.YCB_CLASSES.get(ycb_id, "unknown")
            if self.object_pc_type == 'random':
                indices = torch.randperm(self.num_points)[:self.num_points]
                object_pc = self.object_pcs[obj_name][indices]
                object_pcs.append(object_pc)
            elif self.object_pc_type == 'fixed':
                object_pc = self.object_pcs[obj_name]
                object_pcs.append(object_pc)
            else:
                raise ValueError(f"Unknown object_pc_type: {self.object_pc_type}")
        return object_pcs

# Visualization and Example Usage Functions

def visualize_pointcloud(npz_file):
    """
    Load and visualize point cloud data from a .npz file.

    Args:
        npz_file (str): Path to the .npz file.
    """
    data = np.load(npz_file)
    
    if 'points' not in data:
        raise ValueError(f"File {npz_file} does not contain 'points' key.")
    points = data['points']
    
    print(f"Loaded {len(points)} points from {npz_file}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def load_and_visualize_example(dataset: DexYCB_Dataset, index: int = 0):
    """
    Loads and visualizes a specific example from the dataset.

    Args:
        dataset (DexYCB_Dataset): The dataset instance.
        index (int, optional): Index of the sample to visualize. Defaults to 0.
    """
    sample = dataset[index]
    print(f"Sample keys: {sample.keys()}")

    # Visualize object point clouds
    object_pcs = sample["object_pcs"]
    for i, object_pc in enumerate(object_pcs):
        print(f"Visualizing object {i} point cloud...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_pc.numpy())
        o3d.visualization.draw_geometries([pcd])

    # Visualize MANO hand mesh if needed
    # Assuming you have a method to get MANO mesh from hand_pose
    # mano_mesh = sample["mano_mesh"][0]
    # o3d.visualization.draw_geometries([mano_mesh])

def main(dexycb_dir: str="/home/lightcone/workspace/DRO-retarget/Noise-learn/data/dex_ycb", batch_size: int = 1, filter_objects: list = []):
    """
    Main function to initialize the dataset and perform basic operations.

    Args:
        dexycb_dir (str): Path to the DexYCB dataset directory.
        batch_size (int, optional): Batch size for the dataset. Defaults to 1.
        filter_objects (list, optional): List of object names to filter. Defaults to [].
    """
    dataset = DexYCB_Dataset(
        batch_size=batch_size,
        data_dir=dexycb_dir,
        hand_type="right",
        filter_objects=filter_objects,
        is_train=True,
        num_points=5000,
        object_pc_type='random',
        noise_level='pointcloud',
        mano_root="/home/lightcone/workspace/DRO-retarget/Noise-learn/thirdparty/manopth"
    )
    print(f"Dataset length: {len(dataset)}")

    # Count object occurrences
    ycb_names = []
    for i, data in enumerate(dataset):
        ycb_ids = data["ycb_ids"]
        ycb_names.extend([dataset.YCB_CLASSES.get(ycb_id, "unknown") for ycb_id in ycb_ids])
        if i >= 100:  # Limit to first 100 samples for counting
            break

    counter = Counter(ycb_names)
    print("Object counts:", counter)

    # Visualize the first sample
    load_and_visualize_example(dataset, index=0)

if __name__ == "__main__":
    main()
