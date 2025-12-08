# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" URDF import module.
"""

from collections import OrderedDict
import numpy as np

import newton
from newton._src.utils.import_urdf import parse_urdf as newton_parse_urdf


# Ref: https://github.com/mmatl/urdfpy/blob/master/urdfpy/utils.py
def matrix_to_rpy(R, solution=1):
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.

    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    solution : int
        Either 1 or 2, indicating which solution to return.

    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R, dtype=np.float64)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2, 0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0, 1], R[0, 2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2, 0])
        else:
            p = np.pi + np.arcsin(R[2, 0])
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))

    return np.array([r, p, y], dtype=np.float64)


# Ref: https://github.com/mmatl/urdfpy/blob/master/urdfpy/utils.py
def matrix_to_xyz_rpy(matrix):
    """Convert a 4x4 homogenous matrix to xyzrpy coordinates.

    Parameters
    ----------
    matrix : (4,4) float
        The homogenous transform matrix.

    Returns
    -------
    xyz_rpy : (6,) float
        The xyz_rpy vector.
    """
    xyz = matrix[:3, 3]
    rpy = matrix_to_rpy(matrix[:3, :3])
    return np.hstack((xyz, rpy))


class UrdfInfo:
    def __init__(self, cspace_names, link_index_map, cspace2link, link2cspace, cspace_joint_limits):
        self.cspace_names = cspace_names
        self.link_index_map = link_index_map
        self.cspace2link = cspace2link
        self.link2cspace = link2cspace
        self.cspace_joint_limits = cspace_joint_limits

        self.cspace_name2index_map = {}
        for i, name in enumerate(self.cspace_names):
            self.cspace_name2index_map[name] = i


# def get_link_to_cspace_map(joint_list):
#    """ Just step through the joints and increment whenever the joint isn't fixed.
#    That'll give a map from link to cspace which we can use for lookup.
#    """
#    link_to_cspace_map = []
#    cspace_index = 0
#    for joint in joint_list:
#        if joint.
#    return link_to_cspace_map


def parse_urdf_annotated(filename_or_xml,
                         builder,
                         xform,
                         floating=False,
                         verbose: bool = False):
    newton_parse_urdf(builder, source=filename_or_xml, xform=xform, floating=floating,
                      parse_visuals_as_colliders=False)

    cspace_names = []  # collects the cspace names
    link_index_map = OrderedDict()  # maps from link name -> link index. Also, stores names in link index order.
    cspace2link = []  # for each cspace dim, contains the corresponding link index
    link2cspace = []
    cspace_joint_limits = []

    base_link_name = builder.body_key[0]
    link_index_map[base_link_name] = 0

    for i in range(builder.joint_count):
        joint_key = builder.joint_key[i]
        joint_type = builder.joint_type[i]
        # joint_axis = builder.joint_axis[i]
        # joint_parent_id = builder.joint_parent[i]
        joint_child_id = builder.joint_child[i]
        if joint_type != newton.JointType.FIXED:
            cspace_names.append(joint_key)
            cspace2link.append(joint_child_id)
            joint_dof_id = builder.joint_qd_start[i]
            cspace_joint_limits.append((builder.joint_limit_lower[joint_dof_id], builder.joint_limit_upper[joint_dof_id]))
        link2cspace.append(len(cspace_names) - 1)
        link_index_map[builder.body_key[joint_child_id]] = joint_child_id

    if verbose:
        print("done loading", filename_or_xml if filename_or_xml.endswith(".urdf") else "urdf")

    return UrdfInfo(cspace_names, link_index_map, cspace2link, link2cspace, cspace_joint_limits)
