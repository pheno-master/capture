"""Inverse kinematics through move_group's /compute_ik service."""

from __future__ import annotations

import time

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node

from ..core.ik import ik_seeds, select_joints
from .msg_conversions import matrix_to_pose_fields


class IkClient:
    def __init__(self, node: Node, planning_group: str, ee_link: str, base_frame: str,
                 arm_joints: tuple[str, ...], attempts: int = 10, timeout_s: float = 0.05,
                 joint_limit_rad: float = 2.6, seed: int = 0) -> None:
        self._node = node
        self._group = planning_group
        self._ee_link = ee_link
        self._base = base_frame
        self._joints = arm_joints
        self._attempts = attempts
        self._timeout_s = timeout_s
        self._limit = joint_limit_rad
        self._rng = np.random.default_rng(seed)
        self._client = node.create_client(GetPositionIK, "compute_ik")

    def wait_for_service(self, timeout_s: float) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_s)

    def solve(self, T_world_ee: np.ndarray) -> dict[str, float] | None:
        """Arm joint positions that put ee_link at T_world_ee (collision-free),
        or None if no attempt found one."""
        for seed in ik_seeds(self._joints, self._attempts, self._limit, self._rng):
            response = self._call(self._request(T_world_ee, seed))
            if response is not None and response.error_code.val == MoveItErrorCodes.SUCCESS:
                js = response.solution.joint_state
                return select_joints(dict(zip(js.name, js.position)), self._joints)
        return None

    def _request(self, T_world_ee: np.ndarray, seed: dict[str, float] | None) -> GetPositionIK.Request:
        fields = matrix_to_pose_fields(T_world_ee)
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self._group
        ik.ik_link_name = self._ee_link
        ik.avoid_collisions = True
        ik.pose_stamped.header.frame_id = self._base
        p, q = ik.pose_stamped.pose.position, ik.pose_stamped.pose.orientation
        p.x, p.y, p.z = fields["position"]
        q.x, q.y, q.z, q.w = fields["orientation"]
        sec = int(self._timeout_s)
        ik.timeout = DurationMsg(sec=sec, nanosec=int((self._timeout_s - sec) * 1e9))
        if seed is None:
            ik.robot_state.is_diff = True  # seed from the current state
        else:
            ik.robot_state.joint_state.name = list(seed)
            ik.robot_state.joint_state.position = list(seed.values())
            ik.robot_state.is_diff = True
        return req

    def _call(self, request, wait_s: float = 2.0):
        future = self._client.call_async(request)
        deadline = time.monotonic() + wait_s
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.005)
        return future.result()
