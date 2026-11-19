"""Entry point: `ros2 run plant_capture scan --config config/scan.yaml`.

This is the composition root: the only place where concrete ROS adapters are
created and wired into the ROS-free core. Everything it calls is tested
separately; this file only connects the parts.
"""

from __future__ import annotations

import argparse
import logging
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ..core.capture import FrameCapturer
from ..core.config import ScanConfig, load_config, scan_dir_name, scan_metadata
from ..core.dataset_writer import DatasetWriter
from ..core.image_io import Cv2ImageWriter
from ..core.motion import RollSearchMover
from ..core.session import ScanSession
from ..core.views import hemisphere_views
from .clock import RosClock
from .image_source import RosImageSource
from .joint_source import RosJointSource
from .moveit_mover import MoveItMover
from .tf_pose_source import TfPoseSource

STARTUP_TIMEOUT_S = 10.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static hemisphere scan of one plant.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true", help="print the planned camera positions and exit")
    return p.parse_args(argv)


def print_plan(cfg: ScanConfig) -> None:
    views = hemisphere_views(cfg.views)
    print(f"{len(views)} planned views (camera positions in {cfg.ros.base_frame}, metres):")
    for i, T in enumerate(views):
        x, y, z = T[:3, 3]
        print(f"  {i:3d}: ({x:+.3f}, {y:+.3f}, {z:+.3f})")


def wait_for(fn, what: str, timeout_s: float, clock: RosClock):
    """Poll fn() until it returns something other than None."""
    deadline = clock.now() + timeout_s
    while clock.now() < deadline:
        value = fn()
        if value is not None:
            return value
        clock.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {what}")


def run_scan(node: Node, cfg: ScanConfig) -> Path:
    from moveit.planning import MoveItPy  # imported here: only needed on the robot

    clock = RosClock(node)
    images = RosImageSource(node, cfg.ros.image_topic, cfg.ros.camera_info_topic)
    poses = TfPoseSource(node, cfg.ros.base_frame, cfg.ros.camera_frame, cfg.ros.tf_timeout_s)
    joints = RosJointSource(node, cfg.ros.joint_states_topic)

    intrinsics = images.wait_for_intrinsics(STARTUP_TIMEOUT_S)
    if intrinsics is None:
        raise RuntimeError(f"no CameraInfo on {cfg.ros.camera_info_topic}")
    T_ee_cam: np.ndarray = wait_for(lambda: poses.static_transform(cfg.ros.ee_link, cfg.ros.camera_frame),
                                    f"hand–eye tf {cfg.ros.ee_link} <- {cfg.ros.camera_frame}",
                                    STARTUP_TIMEOUT_S, clock)

    moveit = MoveItPy(node_name=f"{node.get_name()}_moveit")
    mover = RollSearchMover(
        MoveItMover(moveit, cfg.ros.planning_group, cfg.ros.ee_link, cfg.ros.base_frame, T_ee_cam, node.get_logger()),
        cfg.motion.roll_candidates_deg,
    )
    capturer = FrameCapturer(images, poses, joints, clock, cfg.capture)

    created = datetime.now()
    out_dir = Path(cfg.output_root) / scan_dir_name(cfg.plant_id, created)
    writer = DatasetWriter(out_dir, intrinsics, Cv2ImageWriter(), metadata=scan_metadata(cfg, created))

    summary = ScanSession(mover, capturer, writer, clock).run(hemisphere_views(cfg.views))
    writer.finalize(summary)
    node.get_logger().info(
        f"captured {summary.captured}/{summary.planned} views in {summary.total_time_s:.1f} s -> {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.plan_only:
        print_plan(cfg)
        return

    rclpy.init()
    node = Node("plant_scan")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)  # callbacks run here
    spinner.start()
    try:
        run_scan(node, cfg)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
