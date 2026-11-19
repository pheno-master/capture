# plant_capture

ROS 2 (Jazzy) Python package that moves the JetCobot through a static
hemisphere of camera poses, captures one image per pose with the camera pose
from forward kinematics, and writes a scan folder for the Go reconstruction
(`plantrecon`) and the COLMAP + 3DGS baseline.

## Design

The package is split into a **ROS-free core** and **thin ROS adapters**.

```
src/
  core/                  pure Python + numpy; fully unit-tested
    geometry.py          transforms, look_at, quaternions, hand–eye target
    model.py             data types (Intrinsics, CapturedFrame, FrameRecord, ScanSummary, ...)
    schema.py            the scan-folder format: file names, convention, atomic JSON I/O
    interfaces.py        Protocols: Clock, ImageSource, PoseSource, JointSource, Mover, FrameSink
    views.py             hemisphere view planning (condition A)
    capture.py           FrameCapturer: settle, fresh image, pose at image stamp, sync check
    motion.py            RollSearchMover: tries rolls about the optical axis
    session.py           ScanSession: move → capture → store, with timing
    dataset_writer.py    DatasetWriter (a FrameSink)
    dataset_reader.py    load_dataset
    image_io.py          ImageWriter protocol + OpenCV implementation
    config.py            YAML/JSON → frozen dataclasses (unknown keys rejected)
  ros/                   adapters implementing the core Protocols
    image_source.py      Image + CameraInfo subscriptions
    tf_pose_source.py    camera pose from tf2 (FK + static hand–eye)
    joint_source.py      JointState subscription
    moveit_mover.py      MoveIt 2 (moveit_py) motion
    clock.py             ROS clock
    msg_conversions.py   message ↔ core types (no rclpy import, unit-tested)
    scan_node.py         composition root / entry point
  tools/
    export_colmap.py     scan folder → COLMAP text model with known poses
```

How the principles show up:

- **Single responsibility:** each class does one job (`FrameCapturer` only
  synchronises one frame, `ScanSession` only sequences steps,
  `DatasetWriter` only stores).
- **Open/closed:** new behaviour is added by wrapping, not editing, e.g.
  `RollSearchMover` wraps any `Mover`.
- **Liskov / interface segregation:** small Protocols with one or two methods;
  fakes and ROS adapters are interchangeable.
- **Dependency inversion:** the core depends only on `interfaces.py`;
  `scan_node.py` is the only place that creates ROS objects.
- **DRY:** the on-disk format lives only in `schema.py` (+ `model.py`
  `to_dict`/`from_dict`); all transform maths lives in `geometry.py`; the
  config loader is one generic function for every settings class.

## Conventions

Same as the Go side: metres; row-major 4×4; `T_world_cam` maps camera to
world; camera frame is the **optical** frame (+Z forward, +X right, +Y down);
quaternions are (x, y, z, w) in ROS order. Images are BGR uint8 (OpenCV).

## Requirements on the robot

- `robot_state_publisher` publishing the arm's tf tree.
- A **static transform** `ee_link → camera_optical_frame` from hand–eye
  calibration (e.g. `ros2 run tf2_ros static_transform_publisher ...`).
  If your camera driver publishes `camera_link`, add the fixed
  `camera_link → camera_optical_frame` rotation as well.
- The camera driver publishing `Image` and `CameraInfo` with a `plumb_bob`
  distortion model (from `camera_calibration`).
- MoveIt 2 with `moveit_py` and its parameters loaded (normally via a launch
  file using `moveit_configs_utils`).

## Usage

```sh
colcon build --packages-select plant_capture && source install/setup.bash
ros2 run plant_capture scan --config src/plant_capture/config/scan_example.yaml --plan-only
ros2 run plant_capture scan --config src/plant_capture/config/scan_example.yaml
ros2 run plant_capture export_colmap /path/to/scan_dir /path/to/colmap/sparse/0
```

Then check the scan with the Go tool before anything else:

```sh
projcheck -data /path/to/scan_dir -bbox minx,miny,minz,maxx,maxy,maxz
```

## Tests

The tests need only numpy, pytest and PyYAML (OpenCV optional), not ROS:

```sh
python3 -m pytest test
```

## Things to check on the Jetson

- `MoveItMover._execute`: the `MoveItPy.execute` signature differs between
  MoveIt releases; keep the call that matches your installation.
- `RosSettings` defaults (`ee_link`, `planning_group`, topics, frames) are
  placeholders: set them to your JetCobot's names in the YAML.
