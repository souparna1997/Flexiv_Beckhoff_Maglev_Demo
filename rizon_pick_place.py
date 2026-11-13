#!/usr/bin/env python3
"""Repeat the Flexiv/maglev pickup sequence until Ctrl+C. Use --dry-run for offline feedback."""
import argparse
from pathlib import Path
import sys

from support.demo import DryRunIO, load_config, run_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config/demo.json')
    parser.add_argument('--cycles', type=int, help='Stop after N cycles; default: repeat until Ctrl+C')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true', help='Send commands to the real OpenSai/Redis stack (default)')
    mode.add_argument('--dry-run', action='store_true', help='Exercise sequence with ideal feedback without hardware')
    args = parser.parse_args()
    if args.cycles is not None and args.cycles < 1:
        parser.error('--cycles must be a positive integer')
    io = None
    try:
        c = load_config(args.config)
        if not args.dry_run:
            if not c['calibrated']:
                raise ValueError('Calibrate config poses and gripper settings, then set calibrated=true before live execution')
            from support.redis_io import RedisIO
            io = RedisIO(c)
        else:
            print('DRY RUN: ideal feedback; no Redis connection or hardware commands.')
            io = DryRunIO(c)
        io.start()
        run_sequence(io, c, cycles=args.cycles)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOPPED: {exc or "interrupted"}', file=sys.stderr)
        if io is not None:
            try:
                io.hold()
            except Exception as hold_error:
                print(f'Could not request measured-pose hold: {hold_error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
