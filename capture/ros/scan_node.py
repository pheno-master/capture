"""Entry point: `ros2 run capture scan --config config/scan.yaml`.

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
from ..core.geometry import camera_target_to_ee_target
from ..core.motion import RollSearchMover, roll_about_optical_axis
from ..core.session import ScanSession
from ..core.views import hemisphere_views
from .clock import RosClock
from .image_source import RosImageSource
from .joint_source import RosJointSource
from .tf_pose_source import TfPoseSource

STARTUP_TIMEOUT_S = 10.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static hemisphere scan of one plant.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true", help="print the planned camera positions and exit")
    p.add_argument("--check-reach", action="store_true",
                   help="test every view with inverse kinematics only (the arm does not move) and exit")
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


def build_ik(node: Node, cfg: ScanConfig):
    """IK client for the move_group mover, or None if no arm joints are configured."""
    r = cfg.ros
    if not r.arm_joints:
        return None
    from .ik_client import IkClient

    ik = IkClient(node, r.planning_group, r.ee_link, r.base_frame, tuple(r.arm_joints),
                  attempts=r.ik_attempts, timeout_s=r.ik_timeout_s)
    if not ik.wait_for_service(STARTUP_TIMEOUT_S):
        raise RuntimeError("no /compute_ik service: is move_group running?")
    return ik


def build_mover(node: Node, cfg: ScanConfig, T_ee_cam: np.ndarray):
    """The robot-specific motion backend, chosen by ros.mover in the config."""
    r = cfg.ros
    if r.mover == "move_group":
        from .move_group_mover import MoveGroupMover

        mover = MoveGroupMover(node, r.planning_group, r.ee_link, r.base_frame, T_ee_cam, ik=build_ik(node, cfg))
        if not mover.wait_for_server(STARTUP_TIMEOUT_S):
            raise RuntimeError("no MoveGroup action server: is move_group (e.g. demo.launch.py) running?")
        return mover
    from moveit.planning import MoveItPy  # only needed for the moveit_py backend

    from .moveit_mover import MoveItMover

    moveit = MoveItPy(node_name=f"{node.get_name()}_moveit")
    return MoveItMover(moveit, r.planning_group, r.ee_link, r.base_frame, T_ee_cam, node.get_logger())


def check_reach(node: Node, cfg: ScanConfig) -> None:
    """Report which views (and which rolls) have an IK solution, without moving.
    The camera's current pose is tested first: it must be reachable, so a
    failure there means a setup problem (frames, joint names), not geometry."""
    if not cfg.ros.arm_joints:
        raise RuntimeError("--check-reach needs ros.arm_joints in the config")
    clock = RosClock(node)
    poses = TfPoseSource(node, cfg.ros.base_frame, cfg.ros.camera_frame, cfg.ros.tf_timeout_s)
    T_ee_cam = wait_for(lambda: poses.static_transform(cfg.ros.ee_link, cfg.ros.camera_frame),
                        f"tf {cfg.ros.ee_link} <- {cfg.ros.camera_frame}", STARTUP_TIMEOUT_S, clock)
    T_now = wait_for(lambda: poses.static_transform(cfg.ros.base_frame, cfg.ros.camera_frame),
                     f"tf {cfg.ros.base_frame} <- {cfg.ros.camera_frame}", STARTUP_TIMEOUT_S, clock)
    ik = build_ik(node, cfg)

    def reachable(T_world_cam: np.ndarray) -> bool:
        return ik.solve(camera_target_to_ee_target(T_world_cam, T_ee_cam)) is not None

    sanity = reachable(T_now)
    print(f"sanity check, current camera pose: {'OK' if sanity else 'FAILED'}")
    if not sanity:
        print("  -> IK cannot even reach the pose the arm is in now: check arm_joints, ee_link, base_frame")
    rolls = cfg.motion.roll_candidates_deg
    views = hemisphere_views(cfg.views)
    print(f"\n{'view':>4}  {'position (m)':^26}  " + "  ".join(f"{r:>5.0f}°" for r in rolls))
    total = 0
    for i, T in enumerate(views):
        hits = [reachable(roll_about_optical_axis(T, r)) for r in rolls]
        total += any(hits)
        x, y, z = T[:3, 3]
        print(f"{i:4d}  ({x:+.3f}, {y:+.3f}, {z:+.3f})  " + "  ".join(f"{'yes' if h else '-':>6}" for h in hits))
    print(f"\n{total}/{len(views)} views reachable with at least one roll")


def run_scan(node: Node, cfg: ScanConfig) -> Path:
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

    mover = RollSearchMover(build_mover(node, cfg, T_ee_cam), cfg.motion.roll_candidates_deg)
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
        if args.check_reach:
            check_reach(node, cfg)
        else:
            run_scan(node, cfg)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
