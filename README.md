# Flexiv Beckhoff Maglev Demo

A repeating Python pick state machine using the same OpenSai/Redis and Flexiv
gripper interface as `sra_project`. The maglev tile is stationary for this first
version; there is no Beckhoff communication or tile motion yet.

```text
Startup: RAISE_TO_CLEARANCE → INIT
First cycle: REACH_MAGLEV_TILE → OPEN_GRIPPER → APPROACH_OBJECT
             → CLOSE_GRIPPER → PICK_UP_OBJECT → MOVE_TO_INIT
Later cycles: REACH_MAGLEV_TILE → LOWER_TO_PLACE → RELEASE_OBJECT
              → RETURN_ABOVE_TILE → APPROACH_OBJECT → CLOSE_GRIPPER
              → PICK_UP_OBJECT → MOVE_TO_INIT → (repeat)
```

At startup, the TCP moves vertically to `clearance_z` (0.2 m), preserving its
measured X/Y position and orientation, before moving to init. After grasping,
`PICK_UP_OBJECT` again moves vertically to the same absolute Z height before
returning to init. The sequence then repeats from `REACH_MAGLEV_TILE` until Ctrl+C.
On the first cycle, the gripper opens at the above-tile pose before the first pickup.
On later cycles, the robot carries the object from init to above the tile, lowers
to the configured grasp pose, and opens the gripper there to place the object.
It waits for open-gripper feedback before returning to the above-tile pose
(currently Z = 0.2 m), then descends to the grasp pose, closes, lifts, and returns
to init again. There is no release at init. Placement must settle before release,
and the return above the tile must settle before the next descent.
Use `--cycles N` to stop at init after N cycles.
Motion states wait for measured position and orientation to remain within tolerance
for `settle_time`. Gripper states hold their Cartesian pose and use SRA-style
width checks relative to the configured targets:

- Opening advances when `width > open_width - width_tolerance`.
- Closing advances when `width < close_width + width_tolerance`.

For example, a closing target of 38 mm with a 1 mm tolerance advances below 39 mm.
A zero closing target retains SRA's original check of width below 1 mm.

Both gripper transitions advance immediately from the width measurement, without
a measured-force or pose-settling condition. Commands use a speed of 0.05 m/s and
force of 60 N. If the width condition does not pass within `gripper_timeout`, the
sequence times out and reports the measured width and required limit. Each state has a timeout.
Motion uses fixed Cartesian targets and OpenSai's internal OTG.

## Files

- `rizon_pick_place.py`: command-line entry point; defaults to live robot execution.
- `config/demo.json`: waypoints, speeds, tolerances, timeouts, and gripper settings.
- `support/demo.py`: startup lift, repeating sequence, validation, and ideal dry-run feedback.
- `support/redis_io.py`: real robot Redis interface.
- `xml_config_files/rizon_real.xml`: real OpenSai configuration adapted from SRA.
- `src/main.cpp` and `CMakeLists.txt`: native OpenSai configuration launcher.
- `scripts/launch.sh`: shell wrapper for starting the XML controller, like SRA.
- `tests/test_demo.py`: offline sequence and failure-path checks.

## Install and try offline

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 rizon_pick_place.py --dry-run --cycles 2
python3 -m unittest discover -s tests -v
```

Dry run connects to no services and supplies ideal feedback. It checks sequence
logic, not reachability, collisions, physical grasping, or robot dynamics.

## Calibrate

The supplied coordinates and gripper settings are **examples**, not calibrated
maglev waypoints. `calibrated: false` prevents live execution until configured.
Edit `config/demo.json`:

1. Set `poses.init` to the desired initial TCP pose.
2. Set `poses.above_tile` to a clear pose above the stationary object.
3. Set `poses.grasp` to the object grasp pose.
4. Set `clearance_z` to the absolute TCP Z height used before init and after grasping
   (0.2 m by default). It must be above the grasp pose. Set `poses.above_tile` to
   the desired retreat pose after drop-off; its supplied Z is also 0.2 m.
5. Set gripper opening, closing target, commanded force, speed, and width
   tolerance for the installed fingers and object.
6. Review speeds, tolerances, and timeouts, then set `calibrated` to `true`.

Positions and `clearance_z` are meters in the robot base frame. Orientations
are 3×3 rotation matrices, with the same convention as SRA. The controlled point
is `custom_grasp_tcp` from the SRA robot model. Confirm its tool transform matches
the installed tool. Gripper widths are meters, speeds m/s, and forces N.
`orientation_tolerance_deg` is degrees; timeouts and settling times are seconds.

Both vertical moves retain the measured X/Y position and orientation and target
the absolute `clearance_z`, rather than adding a relative lift distance. Init is
reached only after the vertical move has settled. Check that all connecting paths
have clearance; there is no collision planner. With unequal grasp and above-tile XY coordinates,
the approach is a diagonal Cartesian move, so align them for vertical approach.

## Build the OpenSai launcher

Use the already installed/built OpenSai packages from the SRA environment:

```bash
cmake -S . -B build
cmake --build build -j2
```

The default model directory is the sibling project's
`../sra_project/sai_analysis/urdf/rizon4s`. This deliberately reuses the existing
robot model and meshes rather than copying the entire SRA project. The model
loaded by `rizon_real.xml` is
`flexiv_Rizon4s_kinematics_with_grav_static.urdf` in that directory, selected through
the CMake setting `DEMO_ROBOT_FOLDER`. To relocate it:

```bash
cmake -S . -B build -DDEMO_ROBOT_FOLDER=/absolute/path/to/rizon4s
```

If CMake cannot find the Sai packages, use the same `CMAKE_PREFIX_PATH` as your
OpenSai/SRA build. This launcher loads `rizon_real.xml`, starts in the measured
joint hold controller, and exposes the native OpenSai interface. Do not run a
second OpenSai controller instance against the same robot.

## Run on the robot

Use the same real robot setup as SRA: Redis, the Flexiv Redis driver **with the
gripper**, and this project's OpenSai controller. Keep the workspace clear and
have the emergency stop accessible.

1. Start Redis if it is not already running (`redis-server`).
2. Start the existing Flexiv Redis driver with your established robot driver
   configuration. The supplied demo uses `Rizon4s-063671`, matching the driver's
   `config_rizon4s-063671.xml`. The driver's `robotName`, the XML `robotName`, and
   `config/demo.json`'s `robot_name` must all match.
3. In another terminal, from this folder, start the native controller:

   ```bash
   sh scripts/launch.sh xml_config_files/rizon_real.xml
   ```

   The wrapper runs `./build/maglev_opensai` in the foreground; Ctrl+C stops
   that controller. Redis and the Flexiv driver remain separately managed.
   This command holds the measured joint pose; it does not run the pickup cycle.
   `Webui HTML file has been created successfully` only reports UI generation.

4. After calibrating the demo configuration, start the repeating pickup sequence:

   ```bash
   source .venv/bin/activate
   python3 rizon_pick_place.py
   ```

The plain command runs live. `--live` remains an explicit equivalent, and
`--dry-run` selects offline feedback. The gripper state transitions use measured
width, including when demonstrating without an object.

Ctrl+C requests a measured-pose hold and stops the sequence. For a bounded run,
use `python3 rizon_pick_place.py --cycles 1`. Offline runs can also be bounded,
for example `python3 rizon_pick_place.py --dry-run --cycles 2`.

An alternative config can be selected using `--config /path/to/demo.json`.
The script checks that Redis reports `rizon_real.xml` and a running controller
for `Rizon4s-063671`, requests the Cartesian controller, and waits for its task
joint feedback to match the robot joints.
It uses the existing `opensai::...::Rizon4s-063671` keys and writes gripper commands as
`[width, speed, force]`.

If the robot remains in the driver's idle/floating mode, check the matching keys:

```bash
redis-cli MGET opensai::controllers::Rizon4s-063671::is_running opensai::controllers::Rizon4s-063671::active_controller_name opensai::sensors::Rizon4s-063671::joint_positions
```

After native controller startup, expect `1`, `joint_controller`, and measured
joint positions. Starting `rizon_pick_place.py` selects `cartesian_controller`
and prints `RAISE_TO_CLEARANCE`, then `INIT`, followed by the repeating pickup states.
An XML/driver name mismatch can
leave OpenSai operating on unrelated Redis keys while the driver stays idle.

On timeout, invalid feedback, or Ctrl+C, the script attempts a measured-pose hold
and exits with an error. It leaves the gripper commanded as-is so it does not
release a held object. That hold is a Cartesian goal update through OTG, not a
hardware emergency stop. If Redis is unavailable or the controller changed, the
hold cannot be guaranteed; the script reports that failure. The inherited Redis
feedback has no per-sample timestamp, so this script cannot reliably detect a
publisher that has stopped while leaving plausible old values. Keep the existing
driver/controller watchdogs enabled.

No real robot commands are sent by the offline tests or explicit `--dry-run`.
