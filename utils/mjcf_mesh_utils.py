import os
import trimesh
import numpy as np
from xml.etree import ElementTree as ET


def load_link_geometries_mjcf(robot_name, mjcf_path, link_names, collision=False):
    """
    Load geometries (trimesh objects) for specified links from an MJCF file, considering origins.
    
    Args:
        robot_name (str): Name of the robot.
        mjcf_path (str): Path to the MJCF file.
        link_names (list): List of link names to load geometries for.
        collision (bool): Whether to use collision geometry (if available).
    
    Returns:
        dict: A dictionary mapping link names to their trimesh geometries.
    """
    mjcf_dir = os.path.dirname(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    link_geometries = {}

    for body in root.findall(".//body"):
        body_name = body.attrib.get("name", None)
        if body_name in link_names:
            geom_index = "geom" if not collision else "collision"
            link_mesh = []

            for geom in body.findall(geom_index):
                try:
                    mesh_filename = geom.attrib.get("mesh")
                    xyz = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ")
                    rotation = np.fromstring(geom.attrib.get("quat", "1 0 0 0"), sep=" ")  # Quaternion format

                    if mesh_filename:  # Load mesh file
                        full_mesh_path = os.path.join(mjcf_dir, mesh_filename)
                        mesh = as_mesh(trimesh.load(full_mesh_path))
                        scale = np.fromstring(geom.attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        mesh = apply_transform(mesh, xyz, rotation)
                        link_mesh.append(mesh)
                    else:  # Handle primitive shapes
                        mesh = create_primitive_mesh(geom, xyz, rotation)
                        scale = np.fromstring(geom.attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        link_mesh.append(mesh)
                except Exception as e:
                    print(f"Failed to load geometry for {body_name}: {e}")

            if len(link_mesh) == 0:
                continue
            elif len(link_mesh) > 1:
                link_trimesh = as_mesh(trimesh.Scene(link_mesh))
            elif len(link_mesh) == 1:
                link_trimesh = link_mesh[0]

            link_geometries[body_name] = link_trimesh

    return link_geometries


# Helper functions
def as_mesh(scene_or_mesh):
    """
    Convert a trimesh.Scene into a single trimesh.Trimesh object if necessary.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene_or_mesh.dump())
    else:
        mesh = scene_or_mesh
    return mesh


def apply_transform(mesh, xyz, rotation):
    """
    Apply position and rotation to a mesh.
    """
    transform = np.eye(4)
    transform[:3, 3] = xyz
    transform[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3]
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geom, xyz, rotation):
    """
    Create a primitive mesh (e.g., box, cylinder, sphere, capsule) based on MJCF geometry.
    """
    if geom.tag == "geom":
        geom_type = geom.attrib.get("type", "box")
        size = np.fromstring(geom.attrib.get("size", "0.1 0.1 0.1"), sep=" ")

        if geom_type == "box":
            mesh = trimesh.creation.box(extents=size * 2)
        elif geom_type == "sphere":
            mesh = trimesh.creation.icosphere(radius=size[0])
        elif geom_type == "cylinder":
            mesh = trimesh.creation.cylinder(radius=size[0], height=size[1] * 2)
        elif geom_type == "capsule":
            # Capsule: size[0] is radius, size[1] is half-length
            mesh = trimesh.creation.capsule(radius=size[0], height=size[1] * 2)
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

        mesh = apply_transform(mesh, xyz, rotation)
        return mesh
    else:
        raise ValueError(f"Unsupported geometry tag: {geom.tag}")
