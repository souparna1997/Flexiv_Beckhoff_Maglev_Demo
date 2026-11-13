"""Redis contract copied from the local SRA/OpenSai Flexiv interface."""
import json
import time

import numpy as np

from support.demo import rotation


class RedisIO:
    def __init__(self, c):
        import redis
        self.c = c
        self.client = redis.Redis(**c['redis'], decode_responses=True,
                                  socket_timeout=1, socket_connect_timeout=1)
        root = f"opensai::controllers::{c['robot_name']}"
        self.task = root + '::cartesian_controller::cartesian_task::'
        self.active = root + '::active_controller_name'
        self.running = root + '::is_running'
        self.task_q = root + '::cartesian_controller::joint_task::current_position'
        self.sensor = f"opensai::sensors::{c['robot_name']}::"
        self.command = f"opensai::commands::{c['robot_name']}::gripper::parameters"
        self.config_key = '::sai-interfaces-webui::config_file_name'
        self.engaged = False

    def check(self):
        config, running, controller = self.client.mget(self.config_key, self.running, self.active)
        self.check_controller_status(config, running)
        if controller != 'cartesian_controller':
            raise RuntimeError(f'Cartesian controller is no longer selected: {controller!r}')

    def check_controller_status(self, config, running):
        if config != self.c['expected_config']:
            raise RuntimeError(f"Expected {self.c['expected_config']!r}, found {config!r}")
        if running != '1':
            raise RuntimeError(
                f"OpenSai controller for {self.c['robot_name']!r} is not running "
                f"({self.running}={running!r}). Start scripts/launch.sh first and "
                "make sure the XML robotName matches the Flexiv driver's robotName.")

    @staticmethod
    def array(raw, size):
        if raw is None:
            raise RuntimeError('Missing robot feedback in Redis')
        try:
            result = np.asarray(json.loads(raw), dtype=float).reshape(-1)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f'Invalid robot feedback: {raw!r}') from exc
        if result.size != size or not np.isfinite(result).all():
            raise RuntimeError('Robot feedback has incorrect size or nonfinite values')
        return result

    def pose(self):
        p, r = self.client.mget(self.task + 'current_position', self.task + 'current_orientation')
        return self.array(p, 3), rotation(self.array(r, 9).reshape(3, 3))

    def gripper(self):
        w, f = self.client.mget(self.sensor + 'gripper::width', self.sensor + 'gripper::grasp_force')
        return float(self.array(w, 1)[0]), float(self.array(f, 1)[0])

    def start(self):
        config, running = self.client.mget(self.config_key, self.running)
        self.check_controller_status(config, running)
        self.pose()
        self.gripper()
        # Request selection, then check task joint feedback against measured joints.
        # The active-controller key is a request, not a native acknowledgement.
        self.client.set(self.active, 'cartesian_controller')
        self.engaged = True
        started = time.monotonic()
        time.sleep(0.1)
        while True:
            self.check()
            raw_q, raw_task_q = self.client.mget(self.sensor + 'joint_positions', self.task_q)
            q, task_q = self.array(raw_q, 7), self.array(raw_task_q, 7)
            self.pose()
            if np.max(np.abs(q - task_q)) < 0.002:
                break
            if time.monotonic() - started >= self.c['startup_timeout']:
                raise TimeoutError('Cartesian task feedback did not match measured joints')
            time.sleep(0.01)
        self.client.mset({self.task + 'otg_enabled': '1',
                          self.task + 'force_space_dimension': '0',
                          self.task + 'moment_space_dimension': '0'})

    def command_pose(self, pose, speed):
        p, r = pose
        self.client.mset({self.task + 'goal_position': json.dumps(p.tolist()),
                          self.task + 'goal_orientation': json.dumps(r.tolist()),
                          self.task + 'otg_max_linear_velocity': str(speed)})

    def command_gripper(self, width):
        g = self.c['gripper']
        self.client.set(self.command, json.dumps([width, g['speed'], g['force']]))

    def hold(self):
        if self.engaged:
            # Do not overwrite another controller after a config/controller change.
            self.check()
            self.command_pose(self.pose(), self.c['approach_speed'])
