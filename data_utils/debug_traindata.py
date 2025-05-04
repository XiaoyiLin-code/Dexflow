import torch
import numpy as np
from Train_data import TrainDataset
import plotly.graph_objects as go

def visualize_transformed_links_and_object_pc(robot_pc_initial,robot_pc_target=None,  object_pc=None):
    """
    Visualize the transformed robot links' point cloud (target and initial)
    and object point cloud using Plotly.

    Args:
        robot_pc_target (torch.Tensor): Target robot point cloud with shape (N, 4),
        where the last column is the link index.
        robot_pc_initial (torch.Tensor): Initial robot point cloud with shape (N, 4),
        where the last column is the link index.
        object_pc (torch.Tensor, optional): Object point cloud with shape (M, 3).
    """
    # Convert target robot point cloud to numpy


    # Convert initial robot point cloud to numpy
    robot_pc_initial = robot_pc_initial.detach().cpu().numpy()
    robot_initial_x, robot_initial_y, robot_initial_z = robot_pc_initial[:, 0], robot_pc_initial[:, 1], robot_pc_initial[:, 2]
    robot_initial_link_indices = robot_pc_initial[:, 3]

    # Create a Plotly figure
    fig = go.Figure()
    if robot_pc_target is not None:
        robot_pc_target = robot_pc_target.detach().cpu().numpy()
        robot_target_x, robot_target_y, robot_target_z = robot_pc_target[:, 0], robot_pc_target[:, 1], robot_pc_target[:, 2]
        robot_target_link_indices = robot_pc_target[:, 3]
        # Plot target robot links' point cloud
        unique_links_target = sorted(set(robot_target_link_indices))
        robot_colors_target = [
            'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'pink', 'yellow'
        ]  # Color palette for robot links

        for i, link_index in enumerate(unique_links_target):
            link_mask = (robot_target_link_indices == link_index)  # Mask for the current link
            fig.add_trace(go.Scatter3d(
                x=robot_target_x[link_mask],
                y=robot_target_y[link_mask],
                z=robot_target_z[link_mask],
                mode='markers',
                marker=dict(size=2, color=robot_colors_target[i % len(robot_colors_target)]),
                name=f'Target Link {int(link_index)}'
            ))

    # Plot initial robot links' point cloud
    unique_links_initial = sorted(set(robot_initial_link_indices))
    robot_colors_initial = [
        'darkred', 'darkgreen', 'darkblue', 'darkorange', 'indigo', 'darkcyan', 'purple', 'gold'
    ]  # Different color palette for initial robot links

    for i, link_index in enumerate(unique_links_initial):
        link_mask = (robot_initial_link_indices == link_index)  # Mask for the current link
        fig.add_trace(go.Scatter3d(
            x=robot_initial_x[link_mask],
            y=robot_initial_y[link_mask],
            z=robot_initial_z[link_mask],
            mode='markers',
            marker=dict(size=2, color=robot_colors_initial[i % len(robot_colors_initial)]),
            name=f'Initial Link {int(link_index)}'
        ))

    # Plot object point cloud (if provided)
    if object_pc is not None:
        # Convert object point cloud to numpy
        object_pc = object_pc.detach().cpu().numpy()
        object_x, object_y, object_z = object_pc[:, 0], object_pc[:, 1], object_pc[:, 2]

        # Add object point cloud as a separate trace
        fig.add_trace(go.Scatter3d(
            x=object_x,
            y=object_y,
            z=object_z,
            mode='markers',
            marker=dict(size=3, color='black'),
            name='Object Point Cloud'
        ))

    # Configure the layout
    fig.update_layout(
        title="Robot and Object Point Cloud (Target and Initial)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Show the figure
    fig.show()


def check_and_visualize_plotly(dataset):
    """
    Samples a batch from the dataset and visualizes key point clouds using Plotly.
    Marks the center of each point cloud as red and outputs its position.

    Args:
        dataset (torch.utils.data.Dataset): The dataset object to sample from.
    """
    sample = dataset[0]  # Get the first sample
    
    # Extract relevant data
    robot_pc_target = sample['robot_pc_target'][0]  # Shape: (N, 4), includes link indices in the 4th column
    robot_pc_initial = sample['robot_pc_initial'][0]  # Shape: (N, 4), includes link indices in the 4th column
    object_pc = sample['object_pc'][0]  # Shape: (M, 3)

    # Visualize all point clouds together with Plotly
    visualize_transformed_links_and_object_pc(
        robot_pc_initial,
        object_pc=object_pc
    )


# Assuming your dataset is initialized as follows
batch_size = 1

dataset = TrainDataset(
    batch_size=batch_size,
    robot_name='shadowhand',
    is_train=True,
    debug_object_names=None,
    num_points=5000,
    object_pc_type='random'
)

check_and_visualize_plotly(dataset)
