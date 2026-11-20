"""Mover backed by MoveIt 2's Python API (moveit_py).

This is the only module that talks to MoveIt. If your JetCobot setup already
moves the arm another way (a MoveGroup action client, a vendor driver), write
another class with the same `move_camera_to` method and swap it in
scan_node.py: nothing else changes.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseStamped

from ..core.geometry import camera_target_to_ee_target
from .msg_conversions import matrix_to_pose_fields


class MoveItMover:
    def __init__(self, moveit, planning_group: str, ee_link: str, base_frame: str,
                 T_ee_cam: np.ndarray, logger) -> None:
        """`moveit` is a moveit.planning.MoveItPy instance; T_ee_cam is the
        hand–eye transform (camera pose in the ee_link frame)."""
        self._moveit = moveit
        self._arm = moveit.get_planning_component(planning_group)
        self._group = planning_group
        self._ee_link = ee_link
        self._base_frame = base_frame
        self._T_ee_cam = T_ee_cam
        self._log = logger

    def move_camera_to(self, T_world_cam: np.ndarray) -> bool:
        goal = self._pose_stamped(camera_target_to_ee_target(T_world_cam, self._T_ee_cam))
        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(pose_stamped_msg=goal, pose_link=self._ee_link)
        plan = self._arm.plan()
        if not plan:
            return False
        try:
            self._execute(plan.trajectory)
        except Exception as exc:  # execution errors surface as exceptions in moveit_py
            self._log.error(f"execution failed: {exc}")
            return False
        return True

    def _execute(self, trajectory) -> None:
        # NOTE: MoveItPy.execute's signature changed between MoveIt releases.
        # Older (Humble-era tutorials): execute(trajectory, controllers=[]).
        # Newer: execute(group_name, trajectory, blocking=True).
        # Check `help(MoveItPy.execute)` on the Jetson and keep the matching call.
        try:
            self._moveit.execute(self._group, trajectory, blocking=True)
        except TypeError:
            self._moveit.execute(trajectory, controllers=[])

    def _pose_stamped(self, T_world_ee: np.ndarray) -> PoseStamped:
        fields = matrix_to_pose_fields(T_world_ee)
        msg = PoseStamped()
        msg.header.frame_id = self._base_frame
        p, q = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = fields["position"]
        q.x, q.y, q.z, q.w = fields["orientation"]
        return msg
