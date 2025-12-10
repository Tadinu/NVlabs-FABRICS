# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Standard Library
import os
import time
import copy
import argparse

# Third party
import torch
import numpy as np
import warp as wp

from fabrics_sim.utils.utils import initialize_warp, capture_fabric

# Declare device for fabric
device_int = 0
device = 'cuda:' + str(device_int)

# Set the warp cache directory based on device int
warp_cache_dir = ""
initialize_warp(str(device_int))

# Fabrics imports
from fabrics_sim.fabrics.allegro_pose_fabric import AllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.visualization.robot_visualizer import RobotVisualizer
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

"""
This example demonstrates how to create and use a allegro fabric with fingertip poses.
Additional options include:
    1) graph capture of fabric
    2) rendering of the robot and world
    3) rendering of the fabrics collision spheres
    4) setting batch size (number of robots)

Example usage:
python allegro_pose_fabric_example.py --batch_size=10 --render --cuda_graph
"""

# Reduce print precision
torch.set_printoptions(precision=4)

# Parse arguments
parser = argparse.ArgumentParser(description='Allegro fabric example.')
parser.add_argument('--batch_size', type=int, default=1, help='Specify batch size.')
parser.add_argument('--render', action='store_true', default=True, help='True to render fabric motion.')
parser.add_argument('--vis_col_spheres', action='store_true', default=True,
                    help='True to visualize collision spheres of robot.')
parser.add_argument('--cuda_graph', action='store_true', default=True, help='True to enable graph capture of fabric.')
args = parser.parse_args()

# Settings
use_viz = args.render
render_spheres = args.vis_col_spheres
cuda_graph = args.cuda_graph
batch_size = args.batch_size

# This creates a world model that book keeps all the meshes
# in the world, their pose, name, etc.
print('Importing world')
world_filename = 'allegro_env'
max_objects_per_env = 20
world_model = WorldMeshesModel(batch_size=batch_size,
                               max_objects_per_env=max_objects_per_env,
                               device=device,
                               world_filename=world_filename)

# This reports back handles to the meshes which is consumed
# by the fabric for collision avoidance
object_ids, object_indicator = world_model.get_object_ids()

# Control rate and time settings
control_rate = 60.
timestep = 1. / control_rate
total_time = 60.

# Create Allegro fabric finger poses
ROBOT_INIT_POSE = np.array([0, 0, 0.5, 1, 0, 0, 0])
allegro_fabric = AllegroPoseFabric(batch_size, device, timestep,
                                   robot_base_transform=wp.transform(wp.vec3(ROBOT_INIT_POSE[:3]),
                                                                     wp.quat(ROBOT_INIT_POSE[4],
                                                                             ROBOT_INIT_POSE[5],
                                                                             ROBOT_INIT_POSE[6],
                                                                             ROBOT_INIT_POSE[3])),
                                   graph_capturable=cuda_graph)
num_joints = allegro_fabric.num_joints

# Create integrator for the fabric dynamics.
allegro_integrator = DisplacementIntegrator(allegro_fabric)

# Create starting states for the robot.
q = torch.tensor([0.0, 0.3, 0.3, 0.3,
                  0.0, 0.3, 0.3, 0.3,
                  0.0, 0.3, 0.3, 0.3,
                  0.72383858, 0.60147215, 0.33795027, 0.60845138], device=device)
# Resize according to batch size
q = q.unsqueeze(0).repeat(batch_size, 1).contiguous()
# Start with zero initial velocities and accelerations
qd = torch.zeros(batch_size, num_joints, device=device)
qdd = torch.zeros(batch_size, num_joints, device=device)

# Fingertips targets is (origin, Euler ZYX)
target = np.array([0, 0, 1., 0, 0, 0])
ori_finger_targets = {
    control_frame_name: torch.tensor(target, device=device).expand((batch_size, 6)).float()
    for control_frame_name in AllegroPoseFabric.FINGER_CONTROL_FRAMES
}

# Get body sphere raddi
body_sphere_radii = allegro_fabric.get_sphere_radii()

# Get body sphere locations
sphere_positions_in_fabric, _ = allegro_fabric.get_taskmap("body_points")(q.detach(), None)

# Create visualizer
robot_visualizer = None
if use_viz:
    robot_dir_name = "kuka_allegro"
    robot_name = "allegro"
    suite_vertical_offset = 0.
    robot_visualizer = RobotVisualizer(robot_dir_name, robot_name, batch_size, device,
                                       body_sphere_radii if render_spheres else None,
                                       sphere_positions_in_fabric if render_spheres else None,
                                       ROBOT_INIT_POSE,
                                       world_model, suite_vertical_offset, allegro_fabric.get_joint_names())

# Graph capture
g = None
q_new = None
qd_new = None
qdd_new = None
if cuda_graph:
    # NOTE: elements of inputs must be in the same order as expected in the set_features function
    # of the fabric
    inputs = [ori_finger_targets, "euler_zyx",
              q.detach(), qd.detach(), object_ids, object_indicator]
    g, q_new, qd_new, qdd_new = \
        capture_fabric(allegro_fabric, q, qd, qdd, timestep, allegro_integrator, inputs, device)

# Loop stepping the fabric forward in time while updating targets, and optionally, rendering
start = time.time()
random_finger_targets = {
    control_frame_name: torch.tensor(target, device=device).expand((batch_size, 6)).float()
    for control_frame_name in AllegroPoseFabric.FINGER_CONTROL_FRAMES
}
for i in range(int(control_rate * total_time)):
    # Every two seconds switch targets
    if i % 120 == 0:
        for _, ori_finger_target in ori_finger_targets.items():
            rand_finger_target = random_finger_targets[_]
            random_offset = torch.rand_like(ori_finger_target)
            rand_finger_target[:, :3] = finger_target[:, :3] + 0.1 * random_offset[:, :3]
            rand_finger_target[:, 3:] = random_offset[:, 3:] * 2. * np.pi

    # Save off current joint states for rendering
    q_prev = q.detach()
    qd_prev = qd.detach()

    # Step the fabric forward in time
    if cuda_graph:
        # Replay through the graph with the above changed inputs
        g.replay()

        # Update the fabric states
        q.copy_(q_new)
        qd.copy_(qd_new)
        qdd.copy_(qdd_new)
    else:
        # Set the targets
        allegro_fabric.set_features(random_finger_targets, "euler_zyx",
                                    q.detach(), qd.detach(),
                                    object_ids, object_indicator)

        # Integrate fabrics one step producing new position and velocity.
        q, qd, qdd = allegro_integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)

    # Render, albeit at a lower framerate
    if use_viz and (i % 4 == 0):
        # Get body sphere locations reshape into (batch size x num spheres, 3) tensor
        if render_spheres:
            sphere_positions_in_fabric = allegro_fabric.get_taskmap_position("body_points").detach().cpu()
            sphere_positions_in_fabric = \
                sphere_positions_in_fabric.reshape(batch_size * len(body_sphere_radii), -1).detach().cpu().numpy()
        else:
            sphere_positions_in_fabric = None

        robot_visualizer.drawer.clear_points()
        robot_visualizer.render(q_prev.detach().cpu().numpy(),
                                qd_prev.detach().cpu().numpy() * 0.,  # setting to 0 to avoid jitters
                                sphere_positions_in_fabric,
                                [rand_finger_target.detach().cpu().numpy() for _, rand_finger_target in
                                 random_finger_targets.items()])

    # Get distances to upper, lower joint limits and collision status
    dist_to_upper_limit = allegro_fabric.get_taskmap_position("upper_joint_limit")
    dist_to_lower_limit = allegro_fabric.get_taskmap_position("lower_joint_limit")
    collision = allegro_fabric.collision_status.max().item()

    # Print various signals
    print('time', '%.2f' % (i * timestep),
          'wallclock time', '%.2f' % (time.time() - start),
          'To upper joint limit', '%.3f' % dist_to_upper_limit.min().item(),
          'To lower joint limit', '%.3f' % dist_to_lower_limit.min().item(),
          'Collision', collision,
          'min dist', '%.3f' % allegro_fabric.base_fabric_repulsion.signed_distance.min())

# Destroy visualizer
if use_viz:
    print('Destroying visualizer')
    robot_visualizer.close()

print('Done')
