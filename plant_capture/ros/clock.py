"""Clock backed by the node's ROS clock, so it matches message timestamps."""

from __future__ import annotations

import time

from rclpy.node import Node


class RosClock:
    def __init__(self, node: Node) -> None:
        self._clock = node.get_clock()

    def now(self) -> float:
        return self._clock.now().nanoseconds * 1e-9

    def sleep(self, seconds: float) -> None:
        # Wall-clock sleep is fine on the real robot (ROS time == wall time).
        time.sleep(max(0.0, seconds))
