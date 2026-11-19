"""JointSource backed by sensor_msgs/JointState."""

from __future__ import annotations

import threading
from typing import Mapping

from rclpy.node import Node
from sensor_msgs.msg import JointState

from .msg_conversions import joint_state_to_dict


class RosJointSource:
    def __init__(self, node: Node, topic: str) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, float] = {}
        node.create_subscription(JointState, topic, self._on_joints, 10)

    def _on_joints(self, msg: JointState) -> None:
        update = joint_state_to_dict(msg)
        with self._lock:
            self._latest.update(update)  # some drivers publish joints in groups

    def joint_positions(self) -> Mapping[str, float]:
        with self._lock:
            return dict(self._latest)
