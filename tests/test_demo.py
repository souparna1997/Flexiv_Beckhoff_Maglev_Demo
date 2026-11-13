import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

import rizon_pick_place
from support.demo import DryRunIO, State, load_config, run_sequence
from support.redis_io import RedisIO

CONFIG = Path(__file__).resolve().parents[1] / 'config/demo.json'


class Clock:
    now = 0.

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RecordingIO(DryRunIO):
    def __init__(self, c):
        super().__init__(c)
        self.commands = []
        self.pose_goals = []
        self.gripper_events = []

    def command_pose(self, pose, speed):
        self.pose_goals.append((pose[0].copy(), pose[1].copy(), speed))
        super().command_pose(pose, speed)

    def command_gripper(self, width):
        self.commands.append(width)
        position, orientation = self.pose()
        self.gripper_events.append((width, position.copy(), orientation.copy(), len(self.pose_goals)))
        super().command_gripper(width)


class GripperFeedbackIO(RecordingIO):
    """Supply measured finger motion independently of Cartesian commands."""
    def __init__(self, c, clock, feedback, *, opening=False):
        super().__init__(c)
        self.clock = clock
        self.feedback = feedback
        self.feedback_width = c['gripper']['open_width' if opening else 'close_width']
        self.started = None
        self.pose_commands = []
        self.finger_samples = []

    def command_gripper(self, width):
        super().command_gripper(width)
        if width == self.feedback_width:
            self.started = self.clock.time()

    def gripper(self):
        if self.started is not None and self.commands[-1] == self.feedback_width:
            now = self.clock.time()
            self.fingers = self.feedback(now - self.started)
            self.finger_samples.append((now, *self.fingers))
        return super().gripper()

    def command_pose(self, pose, speed):
        self.pose_commands.append((self.clock.time(), pose[0].copy()))
        super().command_pose(pose, speed)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(CONFIG)
        self.clock = Clock()

    def run_demo(self, robot, *, cycles=1):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_sequence(robot, self.c, self.clock.time, self.clock.sleep, cycles=cycles)
        return output.getvalue()

    def test_full_cycle_order_and_closed_gripper(self):
        robot = RecordingIO(self.c)
        output = self.run_demo(robot).splitlines()
        self.assertEqual(output[:8], ['RAISE_TO_CLEARANCE', 'INIT', 'REACH_MAGLEV_TILE', 'OPEN_GRIPPER',
                                     'APPROACH_OBJECT', 'CLOSE_GRIPPER',
                                     'PICK_UP_OBJECT', 'MOVE_TO_INIT'])
        self.assertEqual(robot.commands, [self.c['gripper']['open_width'], self.c['gripper']['close_width']])
        np.testing.assert_allclose(robot.pose()[0], self.c['poses']['init']['position'])
        self.assertEqual(robot.gripper(), (self.c['gripper']['close_width'], self.c['gripper']['force']))

    def test_no_motion_feedback_times_out(self):
        class Stuck(DryRunIO):
            def command_pose(self, pose, speed):
                pass
        with self.assertRaisesRegex(TimeoutError, 'RAISE_TO_CLEARANCE'):
            self.run_demo(Stuck(self.c))

    def test_startup_lift_preserves_measured_xy_and_orientation(self):
        robot = RecordingIO(self.c)
        start_position = np.array([0.42, -0.13, -0.08])
        start_orientation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        robot.current = (start_position.copy(), start_orientation.copy())
        self.run_demo(robot)
        position, orientation, speed = robot.pose_goals[0]
        np.testing.assert_allclose(position, [start_position[0], start_position[1], self.c['clearance_z']])
        np.testing.assert_allclose(orientation, start_orientation)
        self.assertEqual(speed, self.c['approach_speed'])

    def test_grasp_lift_preserves_measured_xy_and_orientation(self):
        c = self.c

        class ContactDrift(RecordingIO):
            def command_gripper(self, width):
                super().command_gripper(width)
                if width == c['gripper']['close_width']:
                    self.lift_index = len(self.pose_goals)
                    self.contact_position = np.array(c['poses']['grasp']['position']) + [0.005, -0.003, 0.002]
                    self.contact_orientation = np.array(c['poses']['init']['orientation'], dtype=float)
                    self.current = (self.contact_position.copy(), self.contact_orientation.copy())

        robot = ContactDrift(c)
        self.run_demo(robot)
        position, orientation, speed = robot.pose_goals[robot.lift_index]
        np.testing.assert_allclose(position, [*robot.contact_position[:2], c['clearance_z']])
        np.testing.assert_allclose(orientation, robot.contact_orientation)
        self.assertEqual(speed, c['approach_speed'])

    def test_lift_timeout_prevents_return_to_init(self):
        c = self.c

        class StuckLift(RecordingIO):
            def command_pose(self, pose, speed):
                if self.commands and self.commands[-1] == c['gripper']['close_width']:
                    self.pose_goals.append((pose[0].copy(), pose[1].copy(), speed))
                else:
                    super().command_pose(pose, speed)

        robot = StuckLift(c)
        with self.assertRaisesRegex(TimeoutError, 'PICK_UP_OBJECT'):
            self.run_demo(robot)
        np.testing.assert_allclose(robot.pose()[0], c['poses']['grasp']['position'])
        np.testing.assert_allclose(robot.pose_goals[-1][0][:2], c['poses']['grasp']['position'][:2])
        self.assertEqual(robot.pose_goals[-1][0][2], c['clearance_z'])

    def test_two_cycles_place_and_regrasp_after_return_to_init(self):
        robot = RecordingIO(self.c)
        output = self.run_demo(robot, cycles=2).splitlines()
        cycle_states = ['REACH_MAGLEV_TILE', 'OPEN_GRIPPER', 'APPROACH_OBJECT',
                        'CLOSE_GRIPPER', 'PICK_UP_OBJECT', 'MOVE_TO_INIT']
        states = [line for line in output if line in State.__members__]
        later_cycle_states = ['REACH_MAGLEV_TILE', 'LOWER_TO_PLACE', 'RELEASE_OBJECT',
                              'RETURN_ABOVE_TILE', 'APPROACH_OBJECT', 'CLOSE_GRIPPER',
                              'PICK_UP_OBJECT', 'MOVE_TO_INIT']
        self.assertEqual(states, ['RAISE_TO_CLEARANCE', 'INIT'] + cycle_states + later_cycle_states)
        g = self.c['gripper']
        self.assertEqual(robot.commands, [g['open_width'], g['close_width']] * 2)
        self.assertEqual(robot.gripper()[0], g['close_width'])
        self.assertIn('CYCLE 2 COMPLETE: at init.', output)

    def test_later_cycles_release_at_grasp_then_return_above_before_regrasping(self):
        robot = RecordingIO(self.c)
        self.run_demo(robot, cycles=3)
        g = self.c['gripper']
        self.assertEqual(robot.commands, [g['open_width'], g['close_width']] * 3)
        np.testing.assert_allclose(robot.gripper_events[0][1], self.c['poses']['above_tile']['position'])
        for release_index in (2, 4):
            _, release_position, release_orientation, pose_count = robot.gripper_events[release_index]
            np.testing.assert_allclose(release_position, self.c['poses']['grasp']['position'])
            np.testing.assert_allclose(release_orientation, self.c['poses']['grasp']['orientation'])
            above_position, above_orientation, speed = robot.pose_goals[pose_count]
            np.testing.assert_allclose(above_position, self.c['poses']['above_tile']['position'])
            np.testing.assert_allclose(above_orientation, self.c['poses']['above_tile']['orientation'])
            self.assertEqual(speed, self.c['approach_speed'])
            _, grasp_position, _, _ = robot.gripper_events[release_index + 1]
            np.testing.assert_allclose(grasp_position, self.c['poses']['grasp']['position'])

    def test_place_timeout_prevents_opening_gripper(self):
        c = self.c

        class StuckPlace(RecordingIO):
            def command_pose(self, pose, speed):
                if len(self.commands) == 2 and np.allclose(pose[0], c['poses']['grasp']['position']):
                    self.pose_goals.append((pose[0].copy(), pose[1].copy(), speed))
                else:
                    super().command_pose(pose, speed)

        robot = StuckPlace(c)
        with self.assertRaisesRegex(TimeoutError, 'LOWER_TO_PLACE'):
            self.run_demo(robot, cycles=2)
        g = c['gripper']
        self.assertEqual(robot.commands, [g['open_width'], g['close_width']])
        self.assertEqual(robot.gripper()[0], g['close_width'])

    def test_release_timeout_prevents_retreat_and_regrasp(self):
        c = self.c

        class JammedRelease(RecordingIO):
            def command_gripper(self, width):
                super().command_gripper(width)
                if width == c['gripper']['open_width'] and len(self.commands) > 2:
                    self.fingers = (c['gripper']['close_width'], c['gripper']['force'])

        robot = JammedRelease(c)
        with self.assertRaisesRegex(TimeoutError, 'RELEASE_OBJECT'):
            self.run_demo(robot, cycles=2)
        g = c['gripper']
        self.assertEqual(robot.commands, [g['open_width'], g['close_width'], g['open_width']])
        np.testing.assert_allclose(robot.pose()[0], c['poses']['grasp']['position'])

    def test_return_above_tile_timeout_prevents_next_grasp(self):
        c = self.c

        class StuckRetreat(RecordingIO):
            def command_pose(self, pose, speed):
                if len(self.commands) == 3 and np.allclose(pose[0], c['poses']['above_tile']['position']):
                    self.pose_goals.append((pose[0].copy(), pose[1].copy(), speed))
                else:
                    super().command_pose(pose, speed)

        robot = StuckRetreat(c)
        with self.assertRaisesRegex(TimeoutError, 'RETURN_ABOVE_TILE'):
            self.run_demo(robot, cycles=2)
        g = c['gripper']
        self.assertEqual(robot.commands, [g['open_width'], g['close_width'], g['open_width']])
        self.assertEqual(robot.gripper()[0], g['open_width'])

    def test_continuous_sequence_repeats_until_interrupted(self):
        class StopDuringThirdCycle(RecordingIO):
            def check(self):
                if len(self.commands) >= 5:
                    raise KeyboardInterrupt

        robot = StopDuringThirdCycle(self.c)
        with self.assertRaises(KeyboardInterrupt):
            self.run_demo(robot, cycles=None)
        g = self.c['gripper']
        self.assertEqual(robot.commands, [g['open_width'], g['close_width']] * 2 + [g['open_width']])

    def test_close_advances_immediately_from_width_without_force(self):
        g = self.c['gripper']
        robot = GripperFeedbackIO(self.c, self.clock, lambda elapsed: (g['close_width'], 0.))
        self.run_demo(robot)
        lift = np.array(self.c['poses']['grasp']['position'])
        lift[2] = self.c['clearance_z']
        lifted_at = next(t for t, p in robot.pose_commands if t >= robot.started and np.allclose(p, lift))
        self.assertEqual(lifted_at, robot.started)
        self.assertEqual(len(robot.finger_samples), 1)

    def test_close_waits_until_width_is_strictly_below_configured_limit(self):
        g = self.c['gripper']

        def feedback(elapsed):
            if elapsed < 0.1:
                return g['close_width'] + 2 * g['width_tolerance'], g['force']
            if elapsed < 0.2:
                return g['close_width'] + g['width_tolerance'], g['force']
            return g['close_width'] + g['width_tolerance'] / 2, 0.

        robot = GripperFeedbackIO(self.c, self.clock, feedback)
        self.run_demo(robot)
        lift = np.array(self.c['poses']['grasp']['position'])
        lift[2] = self.c['clearance_z']
        lifted_at = next(t for t, p in robot.pose_commands if t >= robot.started and np.allclose(p, lift))
        self.assertGreaterEqual(lifted_at - robot.started, 0.2)
        self.assertEqual(lifted_at, robot.finger_samples[-1][0])

    def test_close_times_out_at_or_above_width_limit_even_with_force(self):
        g = self.c['gripper']
        for width in (g['close_width'] + g['width_tolerance'],
                      g['close_width'] + 2 * g['width_tolerance']):
            with self.subTest(width=width):
                self.clock = Clock()
                robot = GripperFeedbackIO(self.c, self.clock, lambda elapsed: (width, g['force']))
                with self.assertRaisesRegex(TimeoutError, 'CLOSE_GRIPPER.*requires width <'):
                    self.run_demo(robot)
                np.testing.assert_allclose(robot.pose()[0], self.c['poses']['grasp']['position'])

    def test_close_uses_configured_target_for_zero_and_nonzero_widths(self):
        for target, measured in ((0., 0.0001), (0.038, 0.03783)):
            with self.subTest(target=target, measured=measured):
                self.c['gripper']['close_width'] = target
                self.clock = Clock()
                robot = GripperFeedbackIO(self.c, self.clock, lambda elapsed: (measured, 0.))
                self.assertIn('DONE', self.run_demo(robot))
                self.assertEqual(len(robot.finger_samples), 1)

    def test_close_accepts_any_finite_force_when_width_is_closed(self):
        g = self.c['gripper']
        for force in (0., 0.79, g['force'], g['force'] + 1):
            with self.subTest(force=force):
                self.clock = Clock()
                robot = GripperFeedbackIO(
                    self.c, self.clock,
                    lambda elapsed: (g['close_width'], force))
                self.assertIn('DONE', self.run_demo(robot))

    def test_open_advances_immediately_above_sra_limit(self):
        g = self.c['gripper']
        for width in (g['open_width'] - g['width_tolerance'] / 2, g['open_width'] + 0.01):
            with self.subTest(width=width):
                self.clock = Clock()
                robot = GripperFeedbackIO(self.c, self.clock, lambda elapsed: (width, 0.), opening=True)
                self.run_demo(robot)
                approach = np.array(self.c['poses']['grasp']['position'])
                approached_at = next(t for t, p in robot.pose_commands if np.allclose(p, approach))
                self.assertEqual(approached_at, robot.started)
                self.assertEqual(len(robot.finger_samples), 1)

    def test_open_times_out_at_sra_limit(self):
        g = self.c['gripper']
        robot = GripperFeedbackIO(
            self.c, self.clock,
            lambda elapsed: (g['open_width'] - g['width_tolerance'], 0.), opening=True)
        with self.assertRaisesRegex(TimeoutError, 'OPEN_GRIPPER.*requires width >'):
            self.run_demo(robot)
        np.testing.assert_allclose(robot.pose()[0], self.c['poses']['above_tile']['position'])

    def test_open_gripper_requires_feedback(self):
        class Jammed(DryRunIO):
            def command_gripper(self, width):
                self.fingers = (0., 0.)
        with self.assertRaisesRegex(TimeoutError, 'OPEN_GRIPPER'):
            self.run_demo(Jammed(self.c))

    def test_bad_config_rejected(self):
        for key, value in [('travel_speed', -1), ('clearance_z', float('nan')),
                           ('clearance_z', self.c['poses']['grasp']['position'][2])]:
            c = copy.deepcopy(self.c)
            c[key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.json'
                path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_feedback_parser_rejects_missing_nonfinite_and_wrong_size(self):
        for raw in (None, '[NaN]', '[1,2]', 'garbage'):
            with self.assertRaises(RuntimeError):
                RedisIO.array(raw, 1)
        np.testing.assert_equal(RedisIO.array('[0.05]', 1), [0.05])

    def test_python_and_native_controller_use_same_robot(self):
        xml = CONFIG.parents[1] / 'xml_config_files/rizon_real.xml'
        config = ET.fromstring('<config>' + xml.read_text() + '</config>')
        self.assertEqual(config.find('robotControlConfiguration').get('robotName'),
                         self.c['robot_name'])

    def test_stopped_controller_rejected_before_sending_commands(self):
        for running in (None, '0'):
            with self.subTest(running=running):
                robot = RedisIO(self.c)
                robot.client = Mock()
                robot.client.mget.return_value = [self.c['expected_config'], running]
                with self.assertRaisesRegex(RuntimeError, 'is not running'):
                    robot.start()
                robot.client.set.assert_not_called()
                robot.client.mset.assert_not_called()
                self.assertFalse(robot.engaged)

    def test_controller_shutdown_detected_even_when_selection_is_unchanged(self):
        robot = RedisIO(self.c)
        robot.client = Mock()
        robot.client.mget.return_value = [self.c['expected_config'], '1', 'cartesian_controller']
        robot.check()
        robot.client.mget.return_value = [self.c['expected_config'], '0', 'cartesian_controller']
        with self.assertRaisesRegex(RuntimeError, 'is not running'):
            robot.check()

    def test_interrupt_propagates(self):
        class Interrupted(DryRunIO):
            def check(self):
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_demo(Interrupted(self.c))


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(CONFIG)
        self.c['calibrated'] = True

    def test_plain_command_and_live_flag_select_live_io(self):
        for flags in ([], ['--live']):
            with self.subTest(flags=flags):
                with patch('sys.argv', ['rizon_pick_place.py', *flags]), \
                        patch('rizon_pick_place.load_config', return_value=self.c), \
                        patch('support.redis_io.RedisIO') as live, \
                        patch('rizon_pick_place.DryRunIO') as offline, \
                        patch('rizon_pick_place.run_sequence') as sequence:
                    self.assertEqual(rizon_pick_place.main(), 0)
                    live.assert_called_once_with(self.c)
                    live.return_value.start.assert_called_once_with()
                    sequence.assert_called_once_with(live.return_value, self.c, cycles=None)
                    offline.assert_not_called()

    def test_dry_run_selects_offline_io(self):
        with patch('sys.argv', ['rizon_pick_place.py', '--dry-run']), \
                patch('rizon_pick_place.load_config', return_value=self.c), \
                patch('support.redis_io.RedisIO') as live, \
                patch('rizon_pick_place.DryRunIO') as offline, \
                patch('rizon_pick_place.run_sequence') as sequence, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rizon_pick_place.main(), 0)
            live.assert_not_called()
            offline.return_value.start.assert_called_once_with()
            sequence.assert_called_once_with(offline.return_value, self.c, cycles=None)

    def test_cycle_limit_is_forwarded_to_sequence(self):
        with patch('sys.argv', ['rizon_pick_place.py', '--dry-run', '--cycles', '2']), \
                patch('rizon_pick_place.load_config', return_value=self.c), \
                patch('rizon_pick_place.DryRunIO') as offline, \
                patch('rizon_pick_place.run_sequence') as sequence, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rizon_pick_place.main(), 0)
            sequence.assert_called_once_with(offline.return_value, self.c, cycles=2)

    def test_invalid_cycle_limit_is_rejected_before_connecting(self):
        with patch('sys.argv', ['rizon_pick_place.py', '--cycles', '0']), \
                patch('support.redis_io.RedisIO') as live, \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as stopped:
            rizon_pick_place.main()
        self.assertEqual(stopped.exception.code, 2)
        live.assert_not_called()

    def test_plain_command_rejects_uncalibrated_config_before_connecting(self):
        self.c['calibrated'] = False
        with patch('sys.argv', ['rizon_pick_place.py']), \
                patch('rizon_pick_place.load_config', return_value=self.c), \
                patch('support.redis_io.RedisIO') as live, \
                patch('rizon_pick_place.run_sequence') as sequence, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rizon_pick_place.main(), 1)
            live.assert_not_called()
            sequence.assert_not_called()


if __name__ == '__main__':
    unittest.main()
