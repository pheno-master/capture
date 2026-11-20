"""Mover that sends pose goals to a running move_group node through MoveIt's
standard MoveGroup action ("move_action").

Use this when MoveIt is already running (e.g. the robot's demo.launch.py or
move_group.launch.py): it needs no moveit_py and no MoveIt parameters in this
process, only moveit_msgs, which every MoveIt installation provides.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (BoundingVolume, Constraints, MoveItErrorCodes, OrientationConstraint,
                             PositionConstraint)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from ..core.geometry import camera_target_to_ee_target
from .msg_conversions import matrix_to_pose_fields


@dataclass(frozen=True)
class MoveGroupSettings:
    action_name: str = "move_action"
    planning_time_s: float = 5.0
    planning_attempts: int = 5
    velocity_scaling: float = 0.3
    acceleration_scaling: float = 0.3
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = 0.02
    timeout_s: float = 60.0  # planning + execution


class MoveGroupMover:
    def __init__(self, node: Node, planning_group: str, ee_link: str, base_frame: str,
                 T_ee_cam: np.ndarray, settings: MoveGroupSettings = MoveGroupSettings()) -> None:
        self._node = node
        self._group = planning_group
        self._ee_link = ee_link
        self._base = base_frame
        self._T_ee_cam = T_ee_cam
        self._s = settings
        self._client = ActionClient(node, MoveGroup, settings.action_name)

    def wait_for_server(self, timeout_s: float) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def move_camera_to(self, T_world_cam: np.ndarray) -> bool:
        goal = MoveGroup.Goal()
        goal.request = self._request(camera_target_to_ee_target(T_world_cam, self._T_ee_cam))
        goal.planning_options.plan_only = False

        accepted = self._wait(self._client.send_goal_async(goal))
        if accepted is None or not accepted.accepted:
            self._node.get_logger().warning("move_group rejected the goal")
            return False
        result = self._wait(accepted.get_result_async())
        if result is None:
            self._node.get_logger().warning("move_group did not finish in time")
            return False
        code = result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().info(f"move_group failed with error code {code}")
            return False
        return True

    def _request(self, T_world_ee: np.ndarray):
        fields = matrix_to_pose_fields(T_world_ee)
        req = MoveGroup.Goal().request
        req.group_name = self._group
        req.num_planning_attempts = self._s.planning_attempts
        req.allowed_planning_time = self._s.planning_time_s
        req.max_velocity_scaling_factor = self._s.velocity_scaling
        req.max_acceleration_scaling_factor = self._s.acceleration_scaling
        req.start_state.is_diff = True  # start from the current state

        target = Pose()
        target.position.x, target.position.y, target.position.z = fields["position"]
        target.orientation.w = 1.0
        region = BoundingVolume()
        region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[self._s.position_tolerance_m])]
        region.primitive_poses = [target]
        position = PositionConstraint(link_name=self._ee_link, constraint_region=region, weight=1.0)
        position.header.frame_id = self._base

        orientation = OrientationConstraint(link_name=self._ee_link, weight=1.0,
                                            absolute_x_axis_tolerance=self._s.orientation_tolerance_rad,
                                            absolute_y_axis_tolerance=self._s.orientation_tolerance_rad,
                                            absolute_z_axis_tolerance=self._s.orientation_tolerance_rad)
        orientation.header.frame_id = self._base
        q = orientation.orientation
        q.x, q.y, q.z, q.w = fields["orientation"]

        req.goal_constraints = [Constraints(position_constraints=[position],
                                            orientation_constraints=[orientation])]
        return req

    def _wait(self, future):
        """Block until a future completes. Callbacks run on the executor
        thread started in scan_node, so polling here is safe."""
        deadline = time.monotonic() + self._s.timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.01)
        return future.result()
