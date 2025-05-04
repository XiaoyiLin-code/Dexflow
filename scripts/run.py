
import yaml
import argparse
from pathlib import Path

from config.path_config import Path_config
from scripts.retarget.retarget import RetargetWrapper
from scripts.contact_mapping.contact_mapping import ContactMapping
from scripts.squence_method.squence import SquenceRefine_Wrapper
from data_utils.retarget_data_dexycb import DexYCB_Dataset
from retarget_utils.constants import RobotName, HandType,OPERATOR2MANO,SCALE
from other_utils.visualize import *  

class Dexflow:
    """
    Dexflow is a class that provides methods to run a Dexflow pipeline.
    """

    def __init__(self, robot_names, hand_type, data,optimize_type,select_idx):
        self.robot_names = robot_names
        self.hand_type = hand_type
        self.data_dir = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset= build_data(data,select_idx)

        cfgs = build_cfgs(robot_names, hand_type)
        self.retarget_wrapper = RetargetWrapper(
            robot_names=robot_names,
            hand_type=hand_type,
            dataset=self.dataset,
            cfgs=cfgs,
        )
        if optimize_type == "squence":
            self.contact_mapping = ContactMapping(upper_bound=0.0007,
                                                 lower_bound=0.0003,
                                                device=self.device)
                                                      
        self.refine_wrapper = SquenceRefine_Wrapper(
            robot_names=robot_names,
            hand_models=self.retarget_wrapper.hand_models
        )


    def Start_flow(self):
        """
        Starts the Dexflow pipeline.
        """
        all_vertices = []
        for idx, data in enumerate(self.dataset):
            for robot_name in self.robot_names:
                retarget_qpos=self.retarget_wrapper.retarget(data,robot_name)      
                object_pc = data["object_pcs"]*1000/SCALE[robot_name.name]
                object_mesh = data["object_meshes"]
                mano_mesh = data["mano_mesh"] 
                object_mesh_scale =[]
                for mesh in object_mesh:
                    # mesh.vertices = mesh.vertices *1000/SCALE[robot_name.name] 
                    vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32)  # 转为 tensor
                    all_vertices.append(vertices_tensor)
                all_vertices = torch.stack(all_vertices).to(self.device)
                robot_pc = [
                    self.retarget_wrapper.hand_models[robot_name.name].get_transformed_links_gene_pc(
                        torch.tensor(qpos, dtype=torch.float32).to(self.device)
                    )
                    for qpos in retarget_qpos
                ]
                robot_pc_tensor_origin = torch.stack(robot_pc)  # Shape: [B, N_points, 3]

                self.contact_mapping.set_up(
                    robot_pc=robot_pc_tensor_origin,
                    object_pc=object_pc,
                    mask_names=get_mask_names(robot_name),
                    link_names=self.retarget_wrapper.hand_models[robot_name.name].manage_link_names(robot_name.name),
                )
                indices_dict, mask_dict, robot_indices_dict_list=self.contact_mapping.get_contact_map_sqpence()
                optimized_pose=self.refine_wrapper.optimize_finger(
                    robot_name=robot_name,
                    init_qpos=retarget_qpos,
                    object_mesh=object_mesh,
                    object_pcs=object_pc,
                    indices_dict=indices_dict,
                    robot_indices_dict=robot_indices_dict_list,
                )
                robot_pc = [
                    self.retarget_wrapper.hand_models[robot_name.name].get_transformed_links_gene_pc(
                        torch.tensor(qpos, dtype=torch.float32).to(self.device)
                    )
                    for qpos in optimized_pose
                ]
                robot_mesh = [
                    self.retarget_wrapper.hand_models[robot_name.name].get_transformed_links_gene_pc(
                        torch.tensor(qpos, dtype=torch.float32).to(self.device), return_meshes=True
                    )
                    for qpos in optimized_pose
                ]
                robot_pc_tensor = torch.stack(robot_pc)  # Shape: [B, N_points, 3]
                print("object_mesh",object_mesh)
                visualize_and_save_transformed_links_and_object_pc(
                    robot_pc=robot_pc_tensor[0],
                    robot_pc_origin=robot_pc_tensor_origin[0],
                    robot_mesh=robot_mesh[0],                  
                    object_pc=object_pc[0],
                    object_mesh=object_mesh[0],
                    mano_mesh=mano_mesh[0],
                    contact_point_indices=robot_indices_dict_list[0]["all"],
                    object_contact_point_indices=indices_dict[0]["all"]
                )


def load_yaml( path: Path):
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, "r") as file:
        return yaml.safe_load(file)

def get_mask_names(robot_name=RobotName.allegro):
    if robot_name==RobotName.shadow:
        mask_names=["ffdistal","mfdistal","rfdistal","thdistal","lfdistal"]
    elif robot_name==RobotName.allegro:
        mask_names=["link_3.0_tip","link_7.0_tip","link_11.0_tip","link_15.0_tip","None"]
    return mask_names

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for configuring robot hand retargeting."""
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Motion retargeting configuration tool')
    parser.add_argument('-r', '--robot_names', 
                        nargs='+', 
                        required=True,
                        choices=[name.lower() for name in RobotName.__members__], 
                        help='List of robot names, available choices: {}'.format([name.lower() for name in RobotName.__members__]))
    
    parser.add_argument('-ht', '--hand_type',
                        required=True,
                        choices=['right', 'left'],
                        help='Hand type, available choices: right/left')
    parser.add_argument('-d', '--data',
                        required=True,
                        default='dexycb',
                        help='Directory containing the data files')
    parser.add_argument('-ot', '--opt_type',
                        required=True,
                        choices=['squence', 'contact'],
                        help='Optimization type, available choices: squence/contact')
    parser.add_argument('-si', '--select_idx',
                        required=True,
                        type=int,
                        nargs='+',  # 允许接受一个或多个整数
                        help='List of indices to select')

    
    return parser.parse_args()

def build_data(data,select_idx):
    """
    Builds the data directory based on the provided data type.
    
    Args:
        data (str): The type of data to build the directory for.
    
    Returns:
        str: The constructed data directory path.
    """

    if data == 'dexycb':
        return  DexYCB_Dataset(
            batch_size=1,
            select_idx=select_idx,
        )
    elif data == 'dexnet':
        return '/home/yangchen/dexnet'
    else:
        raise ValueError("Invalid data type. Choose either 'dexycb' or 'dexnet'.")

def build_cfgs(robot_names, hand_type):
    return [load_yaml((
        Path(Path_config.BASE_DIR) 
        / "config" 
        / "mapping_config"
        / f"{robot_name.name}_hand_{hand_type.name}.yml"
    )) for robot_name in robot_names]

def main():
    """
    Main function to parse arguments and run the retargeting.
    Handles argument parsing, error handling, and invokes the retargeting process.
    """
    # Parse command-line arguments
    args = parse_args()

    hand_type = HandType[args.hand_type.lower()]
    robot_names = [RobotName[name.lower()] for name in args.robot_names]

    dexflow = Dexflow(robot_names, hand_type, args.data, args.opt_type,args.select_idx)
    dexflow.Start_flow()
   

if __name__ == "__main__":
    main()
