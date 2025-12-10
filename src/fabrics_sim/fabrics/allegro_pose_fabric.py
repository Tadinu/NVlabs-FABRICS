# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional
import torch
import warp as wp

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import euler_to_matrix, matrix_to_euler
from fabrics_sim.utils.rotation_utils import quaternion_to_matrix, matrix_to_quaternion


class AllegroPoseFabric(BaseFabric):
    """
    Creates a fabric for the kuka-allegro that opens up a pose action space for the palm
    and PCA'ed action space for the hand. Includes self-collision, env collision avoidance,
    joint limiting, accel/jerk limiting, speed control, redundancy resolution.
    """

    FINGER_CONTROL_FRAMES = ["index_biotac_tip", "middle_biotac_tip",
                             "ring_biotac_tip", "thumb_biotac_tip"]

    def __init__(self, batch_size, device, timestep,
                 robot_base_transform: Optional[wp.transform] = None,
                 graph_capturable=True):
        """
        Constructor. Specifies parameter file and constructs the fabric.
        :param batch_size: size of the batch
        :param device: type str that sets the device for the fabric
        """
        # Load parameters
        fabric_params_filename = "allegro_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         robot_base_transform=robot_base_transform,
                         graph_capturable=graph_capturable)

        # URDF filpath for allegro
        robot_dir_name = "kuka_allegro"
        robot_name = "allegro"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)

        self.load_robot(robot_dir_name, robot_name, batch_size)

        # Going to set a default config for the cspace attractor that gets
        # used until an actual cspace command comes in
        default_config = \
            torch.tensor([0.0, 0.75, 0.75, 0.75,
                          0.0, 0.75, 0.75, 0.75,
                          0.0, 0.75, 0.75, 0.75,
                          1.57, 0.5, 0.5, 0.5], device=self.device)
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        # Construct the fabric.
        self.construct_fabric()

        # Allocate palm pose target tensor (b x (3 + 9))
        # 3 dim for origin target, 12 dim for stacked 3x3 transform target (rx', ry', rx')
        self._finger_pose_targets = {
            control_frame_name: torch.zeros(batch_size, 12, device=device)
            for control_frame_name in self.FINGER_CONTROL_FRAMES
        }

        # Storing the target expressed in the taskspace actually used
        self._native_fingertips_pose_targets = {}

    def add_joint_limit_repulsion(self):
        """
        Adds forcing joint repulsion to the fabric.
        """

        joints = self.urdfpy_robot.joint_map.items()
        # Create upper joint limiting
        # Pulling lower joint limits from urdf
        upper_joint_limits = []
        lower_joint_limits = []
        for joint_name, joint in joints:
            # NOTE: We are only supporting revolute joints right now.
            if joint.type == 'revolute':
                upper_joint_limits.append(joint.limit.upper)
                lower_joint_limits.append(joint.limit.lower)

        # Create upper joint limiting
        # Create taskmap and its container.
        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create geometric fabric term and add to taskmap container.
        is_forcing = True
        fabric_name = "joint_limit_repulsion"
        fabric = JointLimitRepulsion(is_forcing, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, fabric_name, fabric)

        # Create lower joint limiting
        # Create taskmap and its container.
        taskmap_name = "lower_joint_limit"
        taskmap = LowerJointLimitMap(lower_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create geometric fabric term and add to taskmap container.
        is_forcing = True
        fabric_name = "joint_limit_repulsion"
        fabric = JointLimitRepulsion(is_forcing, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, fabric_name, fabric)

    def add_cspace_attractor(self, is_forcing):
        """
        Add a cspace attractors to the fabric.
        -----------------------------
        :param is_forcing: bool, indicates whether the fabric term will be forcing
                           or not (geometric)
        """
        # Create taskmap and its container.
        taskmap_name = "identity"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create fabric term and add to taskmap container.
        if not is_forcing:
            fabric_name = "cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)
            # Add it to container list in the root space
            self.add_fabric(taskmap_name, fabric_name, fabric)
        else:
            fabric_name = "forcing_cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['forcing_cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)

        # Add it to container list in the root space
        self.add_fabric(taskmap_name, fabric_name, fabric)

    def add_finger_points_attractor(self):
        # Set name for taskmap, create it, and add to pool of taskmaps.
        for control_frame_name in self.FINGER_CONTROL_FRAMES:
            taskmap = RobotFrameOriginsTaskMap(self.urdf_path, self.robot_base_transform, [control_frame_name],
                                               self.batch_size, self.device)
            self.add_taskmap(control_frame_name, taskmap, graph_capturable=self.graph_capturable)

            # Create and add geometric attractor
            fabric_name = f'{control_frame_name}_attractor'
            is_forcing = True
            fabric = Attractor(is_forcing, self.fabric_params[fabric_name],
                               self.device, graph_capturable=self.graph_capturable)

            # Add it to container list
            self.add_fabric(control_frame_name, fabric_name, fabric)

    def add_body_repulsion(self):
        """
        Creates body spheres and repulsion between body spheres (self-collision) and also between
        body spheres and environment objects.
        """
        # Create list of frames that will be used to place body spheres at their origins
        collision_sphere_frames = self.fabric_params['body_repulsion']['collision_sphere_frames']

        # List of sphere radii, one for each frame origin
        self.collision_sphere_radii = self.fabric_params['body_repulsion']['collision_sphere_radii']

        assert (len(collision_sphere_frames) == len(self.collision_sphere_radii)), \
            "length of link names does not equal length of radii"

        # Declare which body spheres need to avoid collision
        collision_sphere_pairs = self.fabric_params['body_repulsion']['collision_sphere_pairs']

        # Calculate the body collision matrix
        collision_matrix = torch.zeros(len(collision_sphere_frames), len(collision_sphere_frames), dtype=int,
                                       device=self.device)

        # If frames for collision sphere pairs were not manually specified, then look for the
        # link prefix pairs so that spheres associated with one link can avoid spheres of the other link
        if len(collision_sphere_pairs) == 0:
            # Find links via prefixes to gather collision spheres for self collision avoidance.
            collision_link_prefix_pairs = self.fabric_params['body_repulsion']['collision_link_prefix_pairs']
            frames_for_prefix1 = None
            frames_for_prefix2 = None
            for prefix1, prefix2 in collision_link_prefix_pairs:
                frames_for_prefix1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames_for_prefix2 = [s for s in collision_sphere_frames if prefix2 in s]

                for sphere1 in frames_for_prefix1:
                    for sphere2 in frames_for_prefix2:
                        collision_sphere_pairs.append([sphere1, sphere2])

        for sphere1, sphere2 in collision_sphere_pairs:
            collision_matrix[collision_sphere_frames.index(sphere1), collision_sphere_frames.index(sphere2)] = 1

        # Set name for taskmap, create it, and add to pool of taskmaps.
        taskmap_name = "body_points"
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, self.robot_base_transform, collision_sphere_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create fabric term and add to taskmap container.
        fabric_name = "repulsion"
        is_forcing = True
        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device)
        sphere_radius = sphere_radius.repeat(self.batch_size, 1)
        fabric = BodySphereRepulsion(is_forcing, self.fabric_params['body_repulsion'],
                                     self.batch_size, sphere_radius, collision_matrix, self.device,
                                     graph_capturable=self.graph_capturable)

        # Add it to container list
        self.add_fabric(taskmap_name, fabric_name, fabric)

        # Add geometric body repulsion
        fabric_geom = BodySphereRepulsion(False, self.fabric_params['body_repulsion'],
                                          self.batch_size, sphere_radius, collision_matrix, self.device,
                                          graph_capturable=self.graph_capturable)

        # Add it to container list
        self.add_fabric(taskmap_name, "geom_repulsion", fabric_geom)

        # Create object that constructs base response and signed distance
        self.base_fabric_repulsion = \
            BaseFabricRepulsion(self.fabric_params['body_repulsion'],
                                self.batch_size,
                                sphere_radius,
                                collision_matrix,
                                self.device)

    def add_cspace_energy(self):
        """
        Add a Euclidean cspace energy to the fabric.
        """
        # Add gripper energy.
        taskmap_name = "identity"
        energy_name = "euclidean"
        self.add_energy(taskmap_name, energy_name, EuclideanEnergy(self.batch_size, self._num_joints, self.device))

    def construct_fabric(self):
        """
        Construct the fabric by adding the various geometric, potential, and energy
        components.
        """
        # Add joint limit repulsion
        self.add_joint_limit_repulsion()

        # Add geometric cspace attractor
        self.add_cspace_attractor(False)

        # Add multi-point gripper attractor
        self.add_finger_points_attractor()

        # Add collision avoidance
        self.add_body_repulsion()

        # Add energy
        self.add_cspace_energy()

    def get_finger_pose_target_as_point(self, control_frame_name):
        """
        Get a finger pose target as a target point (no orientation) in the gripper frame
        :param control_frame_name
        ------------------------------------------
        :return gripper_targets: bx(3n) Pytorch tensor, where n is number of
                                 gripper points
        """

        # Fill in targets
        finger_target = torch.zeros(self.batch_size, 3, device=self.device)
        finger_target[:, :3] = self._finger_pose_targets[control_frame_name][:, :3]
        return finger_target

    def get_sphere_radii(self):
        """
        Returns the radii for the body collision spheres.
        ------------------------------------------
        :return collision_sphere_radii: list of floats containing the radii
        """
        return self.collision_sphere_radii

    @property
    def collision_status(self):
        """
        Returns the collision state for each body sphere of the robot
        ------------------------------------------
        :return collision_status: bxn bool Pytorch tensor, b is batch size, n is number of body
                                  spheres
        """
        return self.base_fabric_repulsion.collision_status

    def set_orientation_target(self, in_target: torch.Tensor, out_target: torch.Tensor):
        # First convert palm target orientation from specified convention to rotation matrix
        if orientation_convention == "euler_zyx":
            assert (in_target.shape[1] == 6), \
                "Pose target must be of dimensions (batch_size x 6) with Euler convention"
            out_target[:, 3:] = \
                torch.transpose(euler_to_matrix(
                    in_target[:, 3:]), 1, 2).reshape(self.batch_size, 9)
            # torch.transpose(transforms.euler_angles_to_matrix(
            #    in_target[:, 3:], "ZYX"), 1, 2).reshape(self.batch_size, 9)
        elif orientation_convention == "quaternion":
            assert (in_target.shape[1] == 7), \
                "Pose target must be of dimensions (batch_size x 7) with quaternion convention"
            out_target[:, 3:] = \
                torch.transpose(quaternion_to_matrix(  # transforms.quaternion_to_matrix(
                    in_target[:, [6, 3, 4, 5]]), 1, 2).reshape(self.batch_size, 9)
        else:
            raise ValueError('orientation_convention parameter must be either "euler_zyx" or "quaternion"')

    def set_finger_pose_targets(self, finger_targets: dict[str, torch.Tensor]):
        # Update `self._finger_pose_targets`
        # Insert translational targets into class tensor for holding the target pose
        for finger_part_name, finger_target in self._finger_pose_targets.items():
            in_finger_target = finger_targets[finger_part_name]
            finger_target[:, :3] = finger_targets[:, :3]
            self.set_orientation_target(in_finger_target, finger_target)

        # If multi-point attractor is being used, then convert pose target to targets in the right space
        # from `self._finger_pose_targets`
        for control_frame_name in self.FINGER_CONTROL_FRAMES:
            point_target = self.get_finger_pose_target_as_point(control_frame_name)

            if control_frame_name in self._native_fingertips_pose_targets:
                self._native_fingertips_pose_targets[control_frame_name].copy_(point_target)
            else:
                self._native_fingertips_pose_targets[control_frame_name] = torch.clone(point_target)

            # Pass the gripper target to the gripper attractors and the damping target
            frame_attractor_name = f'{control_frame_name}_attractor'
            try:
                self.fabrics_features[control_frame_name][frame_attractor_name] = \
                    self._native_fingertips_pose_targets[control_frame_name]
                self.get_fabric_term(control_frame_name, frame_attractor_name).damping_position = \
                    self._native_fingertips_pose_targets[control_frame_name]
            except:
                raise ValueError(f'No task map {control_frame_name} or {frame_attractor_name}')

    def set_features(self, in_finger_targets, orientation_convention,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids,
                     object_indicator,
                     cspace_damping_gain=None):
        """
        Passes the input features to the various fabric terms.
        -----------------------------
        :param in_finger_targets: map of finger-part targets, each of which is a bxm Pytorch tensor (origin, rotation),
               where rotation can have 3 elements for Euler "ZYX" angles
               (x_angle, y_angle, z_angle) or 4 elements for quaternion (x, y, z, w)
        :param orientation_convention: str, either "euler_zyx" or "quaternion" (x, y, z, w)
        :param batched_cspace_position: bx7 Pytorch tensor, current fabric position
        :param batched_cspace_velocity: bx7 Pytorch tensor, current fabric velocity
        :param object_ids: 2D int Warp array referencing object meshes
        :param object_indicator: 2D Warp array of type uint64, indicating the presence
                                 of a Warp mesh in object_ids at corresponding index
                                 0=no mesh, 1=mesh
        """
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        # Set pose targets
        self.set_finger_pose_targets(finger_pose_targets)

        # Calculate current location of body sphere origins and their velocity
        body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)

        # Calculate signed distance and repulsion response based on body sphere origin
        # position and velocity and objects in the world
        # NOTE: this calculates both self-collision and robot-world collision response
        self.base_fabric_repulsion.calculate_response(body_point_pos,
                                                      body_point_vel,
                                                      object_ids,
                                                      object_indicator)

        # Pass the collision response data into both the forcing and geometric collision
        # avoidance fabric terms.
        self.fabrics_features["body_points"]["repulsion"] = \
            self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = \
            self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params['cspace_damping']['gain'] = cspace_damping_gain
