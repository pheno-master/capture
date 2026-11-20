"""Mover that sends goals to a running move_group node through MoveIt's
standard MoveGroup action ("move_action").

With an IkClient (the default in scan_node), each camera target is first
converted to arm joint positions with /compute_ik, retrying from several seeds,
and move_group receives a joint goal, which plans far more reliably on small
arms than a pose goal. Without one, it sends a pose goal for ee_link.

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
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from ..core.geometry import camera_target_to_ee_target
from .ik_client import IkClient
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
    joint_tolerance_rad: float = 0.005
    timeout_s: float = 60.0  # planning + execution


class MoveGroupMover:
    def __init__(self, node: Node, planning_group: str, ee_link: str, base_frame: str,
                 T_ee_cam: np.ndarray, settings: MoveGroupSettings = MoveGroupSettings(),
                 ik: IkClient | None = None) -> None:
        self._node = node
        self._group = planning_group
        self._ee_link = ee_link
        self._base = base_frame
        self._T_ee_cam = T_ee_cam
        self._s = settings
        self._ik = ik
        self._client = ActionClient(node, MoveGroup, settings.action_name)

    def wait_for_server(self, timeout_s: float) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def move_camera_to(self, T_world_cam: np.ndarray) -> bool:
        T_world_ee = camera_target_to_ee_target(T_world_cam, self._T_ee_cam)
        if self._ik is not None:
            joints = self._ik.solve(T_world_ee)
            if joints is None:
                self._node.get_logger().info("no IK solution for this view")
                return False
            goal_constraints = self._joint_goal(joints)
        else:
            goal_constraints = self._pose_goal(T_world_ee)
        return self._send(goal_constraints)

    def _send(self, goal_constraints: Constraints) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self._group
        req.num_planning_attempts = self._s.planning_attempts
        req.allowed_planning_time = self._s.planning_time_s
        req.max_velocity_scaling_factor = self._s.velocity_scaling
        req.max_acceleration_scaling_factor = self._s.acceleration_scaling
        req.start_state.is_diff = True  # start from the current state
        req.goal_constraints = [goal_constraints]
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

    def _joint_goal(self, joints: dict[str, float]) -> Constraints:
        tol = self._s.joint_tolerance_rad
        return Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=pos, tolerance_above=tol, tolerance_below=tol, weight=1.0)
            for name, pos in joints.items()
        ])

    def _pose_goal(self, T_world_ee: np.ndarray) -> Constraints:
        fields = matrix_to_pose_fields(T_world_ee)
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
        return Constraints(position_constraints=[position], orientation_constraints=[orientation])

    def _wait(self, future):
        """Block until a future completes. Callbacks run on the executor
        thread started in scan_node, so polling here is safe."""
        deadline = time.monotonic() + self._s.timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.01)
        return future.result()
