"""
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import os
import numpy as np


def quat_from_euler_xyz(roll: np.array, pitch: np.array, yaw: np.array) -> np.array:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qw, qx, qy, qz])


class RobotVisualizer():
    def __init__(self, robot_dir_name, robot_name, batch_size, device,
                 robot_body_sphere_radii, robot_body_sphere_positions,
                 robot_init_pose,
                 world_model, suite_vertical_offset, fabric_joint_names,
                 spacing=3.):

        ## Isaac Sim related imports
        from isaacsim import SimulationApp
        self.simulation_app = SimulationApp({"headless": False})

        from isaacsim.core.api.simulation_context import SimulationContext
        from isaacsim.core.prims import Articulation
        from isaacsim.core.api.world import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import XFormPrim

        from isaacsim.util.debug_draw import _debug_draw
        self.drawer = _debug_draw.acquire_debug_draw_interface()

        from isaacsim.storage.native import get_assets_root_path
        ISAAC_NUCLEUS_USD_PATHS = {
            "allegro": f"{get_assets_root_path()}/Isaac/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd"
            # "allegro": "<path_to_TacEx>/source/tacex_assets/tacex_assets/data/Robots/allegro_hand_gsmini.usd"
        }

        # Fabrics imports
        from fabrics_sim.utils.path_utils import get_world_path, get_object_urdf_path, get_robot_urdf_path, \
            get_robot_usd_path

        self.device = device
        self.robot_init_pose = robot_init_pose
        self.suite_vertical_offset = suite_vertical_offset
        self.batch_size = batch_size
        self.fabric_joint_names = fabric_joint_names

        self.simulation_context = SimulationContext()

        if World.instance():
            World.instance().clear_instance()
        self.world = World()
        self.world.scene.add_default_ground_plane()

        # Add body spheres
        if robot_body_sphere_radii:
            self.num_spheres = len(robot_body_sphere_radii)
            self.robot_sphere_handles = []
            self.add_robot_spheres(robot_body_sphere_radii)

            self.robot_sphere = XFormPrim(prim_paths_expr="/World/Suite_*/Sphere_*",
                                          name='robot_sphere')
            self.robot_sphere.initialize()
            self.world.scene.add(self.robot_sphere)

        # new_positions = np.array([[-1.0, 1.0, 0], [1.0, 1.0, 0]])
        self.suite_base_positions = self.create_suite_grid(spacing=spacing)

        # configure robot model path + add its reference
        robot_path = get_robot_usd_path(robot_dir_name, robot_name)
        if not os.path.exists(robot_path):
            robot_path = ISAAC_NUCLEUS_USD_PATHS[robot_name]

        for i in range(batch_size):
            robot_str = "/World/Suite_" + str(i + 1)
            print('Added suite', str(i + 1))
            add_reference_to_stage(usd_path=robot_path, prim_path=robot_str)

        # Now initialize physics
        self.simulation_context.initialize_physics()

        # Create articulation which we will use to teleport the robot joints
        robot_range = "/World/Suite_[1-9]|[1-9][0-9]{1,3}|9000"
        self.robots = Articulation(prim_paths_expr=robot_range, name="robots")
        self.robots.initialize()
        self.robots.set_enabled_self_collisions(np.array([False] * batch_size))
        self.robots.set_body_disable_gravity(np.zeros((batch_size, self.robots.num_bodies)))

        # Add robot view to world
        self.world.scene.add(self.robots)

        # set root body poses
        self.robots.set_world_poses(positions=np.array([robot_init_pose[:3]] * batch_size) + self.suite_base_positions,
                                    orientations=np.array([robot_init_pose[3:]] * batch_size))

        #        self.robots.set_solver_position_iteration_counts(np.full((self.batch_size,), 64))
        #        self.robots.set_solver_velocity_iteration_counts(np.full((self.batch_size,), 3))

        inertias = np.tile(np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]),
                           (self.batch_size, self.robots.num_bodies, 1))
        self.robots.set_body_inertias(inertias)

        # print(self.robots.get_body_masses())
        # input(self.robots.get_body_inertias())

        # Add in fabrics world model objects
        if world_model is not None:
            self.object_handles = []
            self.add_world_objects(world_model)
            self.objects = XFormPrim(prim_paths_expr="/World/Suite_*/Object_*", name='objects')
            self.objects.initialize()
            self.world.scene.add(self.objects)

        # print(self.robots.dof_names)
        # input(self.robots.get_dof_limits())

        # take a physics step to create all the handles, etc.
        self.simulation_context.play()

        # Since the joint order in isaac sim does not necessarily match the joint order in fabrics,
        # we have to generate a list of joint indices that allow us to rearrange the incoming
        # fabric joint positions such that the joint positions are issued to the correct isaac sim
        # joints
        self.joint_indices = \
            [self.fabric_joint_names.index(joint_name) for joint_name in self.robots.dof_names]

    def add_robot_spheres(self, robot_body_sphere_radii):
        from isaacsim.core.api.objects import VisualSphere

        for i in range(self.batch_size):
            sphere_handles = []
            for j in range(len(robot_body_sphere_radii)):
                sphere_name = '/World/Suite_' + str(i + 1) + '/Sphere_' + str(j + 1)
                sphere_handles.append(
                    self.world.scene.add(
                        VisualSphere(
                            prim_path=sphere_name,
                            name=sphere_name,
                            color=np.array([0, 0, 255]),
                            radius=robot_body_sphere_radii[j])
                    )
                )
            self.robot_sphere_handles.append(sphere_handles)

    def add_world_objects(self, world_model):
        from isaacsim.core.api.objects import VisualCuboid
        for i in range(self.batch_size):
            # Create objects in world
            # Add visualization box for obstacle box to scene.
            object_names = world_model.get_object_names()
            object_handles = []
            for j in range(len(object_names)):
                object_pose_wp = world_model.get_object_transform(object_names[j])  # warp transform
                object_scaling = world_model.get_object_scaling(object_names[j])

                object_name = "/World/Suite_" + str(i + 1) + "/Object_" + str(j + 1)
                position = object_pose_wp[:3].detach().cpu().numpy()
                position += self.suite_base_positions[i]
                orientation = np.zeros(4)
                orientation[0] = object_pose_wp[6]
                orientation[1] = object_pose_wp[3]
                orientation[2] = object_pose_wp[4]
                orientation[3] = object_pose_wp[5]
                object_handles.append(
                    self.world.scene.add(
                        VisualCuboid(
                            prim_path=object_name,
                            name=object_name,
                            translation=position,
                            orientation=orientation,
                            scale=object_scaling.detach().cpu().numpy(),
                            color=np.array([255, 0, 0]))
                    )
                )
            self.object_handles.append(object_handles)

    def set_robot_sphere_positions(self, sphere_positions):
        self.robot_sphere.set_world_poses(positions=sphere_positions)

    #        for i in range(self.batch_size):
    #            for j in range(len(self.robot_sphere_handles[i])):
    #                sphere_pos = sphere_positions[i, j]  + self.suite_base_positions[i]
    #                self.robot_sphere_handles[i][j].set_world_pose(sphere_pos)

    def create_suite_grid(self, spacing=2.):
        # Set up grid spacing

        # Calculate the grid dimensions (approximate square root)
        num_columns = int(np.ceil(np.sqrt(self.batch_size)))
        num_rows = int(np.ceil(self.batch_size / num_columns))

        # Create a grid of positions
        positions = []
        for i in range(num_rows):
            for j in range(num_columns):
                if len(positions) < self.batch_size:
                    x = (j - num_columns // 2) * spacing
                    y = (i - num_rows // 2) * spacing
                    z = self.suite_vertical_offset
                    positions.append([x, y, z])
                else:
                    break

        # Convert to numpy array
        xyz = np.array(positions)

        return xyz

    def render(self, joint_position, joint_velocity, sphere_positions_in_fabric, target_positions_in_world):
        # set the joint positions for each robot
        self.robots.set_joint_positions(joint_position[:, self.joint_indices])
        self.robots.set_joint_velocities(joint_velocity[:, self.joint_indices])

        # draw target
        self.drawer.clear_points()
        for i, target in enumerate(target_positions_in_world):
            pos = (target[:, :3] + self.suite_base_positions).tolist()
            # quat = [self.quat_from_euler(target[i, 3:]) for i in range(target.shape[0])]
            rgba = [0, 1, 0, 1] if len(target_positions_in_world) == 1 else \
                [1 if i == 0 else 0.5 * (i == 3), 1 if i == 0 else 0.8 * (i == 3), i == 2, 1]
            self.drawer.draw_points(pos, [rgba] * len(pos), [60] * len(pos))

        # Move the sphere positions to world frame
        if sphere_positions_in_fabric is not None:
            sphere_world_positions = sphere_positions_in_fabric.copy()
            for i in range(self.batch_size):
                sphere_world_positions[i * self.num_spheres: (i + 1) * self.num_spheres, :] += \
                    np.array([self.suite_base_positions[i]])

            self.set_robot_sphere_positions(sphere_world_positions)

        # step and render
        self.simulation_context.step(render=True)

        # print('errors', joint_position - self.robots.get_joint_positions())

    def close(self):
        self.simulation_context.stop()
        self.simulation_app.close()
