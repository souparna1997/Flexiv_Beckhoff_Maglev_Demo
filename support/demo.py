"""Single-cycle pick sequence; hardware and dry-run share this state machine."""
import json
import math
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np


class State(Enum):
    INIT = auto()
    REACH_MAGLEV_TILE = auto()
    OPEN_GRIPPER = auto()
    APPROACH_OBJECT = auto()
    CLOSE_GRIPPER = auto()
    PICK_UP_OBJECT = auto()
    MOVE_TO_INIT = auto()


def rotation(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (3, 3) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-5)
            or not math.isclose(np.linalg.det(result), 1, abs_tol=1e-5)):
        raise ValueError("Orientation must be a finite 3x3 rotation matrix")
    return result


def load_config(path):
    c = json.loads(Path(path).read_text())
    if type(c['calibrated']) is not bool:
        raise ValueError('calibrated must be a boolean')
    for name in ('init', 'above_tile', 'grasp'):
        pose = c['poses'][name]
        p = np.asarray(pose['position'], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f'{name}: position must contain three finite values')
        rotation(pose['orientation'])
    for key in ('lift_height', 'travel_speed', 'approach_speed', 'position_tolerance',
                'orientation_tolerance_deg', 'settle_time', 'motion_timeout',
                'gripper_timeout', 'startup_timeout', 'loop_hz'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    g = c['gripper']
    for key, value in g.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'gripper.{key} must be finite and nonnegative')
    if (g['open_width'] <= g['close_width'] + g['width_tolerance']
            or min(g['speed'], g['force'], g['width_tolerance']) <= 0):
        raise ValueError('Invalid gripper widths, speed, force, or tolerances')
    if c['poses']['above_tile']['position'][2] <= c['poses']['grasp']['position'][2]:
        raise ValueError('above_tile must be above grasp in base-frame Z')
    return c


def pose_reached(actual, goal, c):
    p, r = actual
    gp, gr = goal
    angle = math.acos(float(np.clip((np.trace(gr.T @ r) - 1) / 2, -1, 1)))
    return (np.linalg.norm(p - gp) <= c['position_tolerance']
            and angle <= math.radians(c['orientation_tolerance_deg']))


def run_sequence(io, c, clock=time.monotonic, sleep=time.sleep):
    poses = {name: (np.array(p['position'], dtype=float), rotation(p['orientation']))
             for name, p in c['poses'].items()}
    lift = (poses['grasp'][0] + [0, 0, c['lift_height']], poses['grasp'][1])
    goals = {State.INIT: poses['init'], State.REACH_MAGLEV_TILE: poses['above_tile'],
             State.OPEN_GRIPPER: poses['above_tile'], State.APPROACH_OBJECT: poses['grasp'],
             State.CLOSE_GRIPPER: poses['grasp'], State.PICK_UP_OBJECT: lift,
             State.MOVE_TO_INIT: poses['init']}
    g = c['gripper']
    for state in State:
        print(state.name, flush=True)
        started, settled = clock(), None
        gripping = state in (State.OPEN_GRIPPER, State.CLOSE_GRIPPER)
        timeout = c['gripper_timeout'] if gripping else c['motion_timeout']
        speed = c['approach_speed'] if state in (State.APPROACH_OBJECT, State.PICK_UP_OBJECT) else c['travel_speed']
        io.command_pose(goals[state], speed)
        if gripping:
            io.command_gripper(g['open_width'] if state == State.OPEN_GRIPPER else g['close_width'])
        while True:
            io.check()
            if gripping:
                width, _ = io.gripper()
                # Match SRA's main pickup: gripper transitions depend only on width.
                if state == State.OPEN_GRIPPER:
                    if width > g['open_width'] - g['width_tolerance']:
                        break
                elif width < g['width_tolerance']:
                    break
            else:
                ready = pose_reached(io.pose(), goals[state], c)
                now = clock()
                settled = (now if settled is None else settled) if ready else None
                if settled is not None and now - settled >= c['settle_time']:
                    break
            now = clock()
            if now - started >= timeout:
                detail = ''
                if gripping:
                    comparison, limit = ('>', g['open_width'] - g['width_tolerance']) if state == State.OPEN_GRIPPER else ('<', g['width_tolerance'])
                    detail = f' (width={width:.6g} m; requires width {comparison} {limit:.6g} m)'
                raise TimeoutError(f'{state.name}: feedback did not satisfy completion conditions within {timeout:g}s{detail}')
            # Reissue fixed goals, including during gripper waits; OpenSai owns the trajectory.
            io.command_pose(goals[state], speed)
            sleep(1 / c['loop_hz'])
    print('DONE: at init. No automatic repeat or release.', flush=True)


class DryRunIO:
    """Ideal feedback exercises sequencing only, not dynamics or collision checking."""
    def __init__(self, c):
        self.c = c
        self.current = (np.zeros(3), np.eye(3))
        self.fingers = (c['gripper']['open_width'], 0.)
        self.states = []

    def start(self):
        pass

    def check(self):
        pass

    def command_pose(self, pose, speed):
        self.current = pose

    def pose(self):
        return self.current

    def command_gripper(self, width):
        if width == self.c['gripper']['open_width']:
            self.fingers = (width, 0.)
        else:
            self.fingers = (width, self.c['gripper']['force'])

    def gripper(self):
        return self.fingers

    def hold(self):
        pass
