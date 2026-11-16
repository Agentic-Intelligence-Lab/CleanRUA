"""Native GPU control test, without a model. Not part of `unittest discover`.

    python tests/smoke_robodojo.py --native-root /path/to/RoboDojo

Scripted calls through the same wrapper the agent uses: capture, hold, lift 1 cm,
close and open the gripper, return home, finish. This checks control, not bowl stacking.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrua.direct_robodojo import RoboDojoWrapper  # noqa: E402
from cleanrua_utils.robodojo_env import RoboDojoEnv  # noqa: E402


def call(name, **arguments):
    return {'name': name, 'arguments': json.dumps(arguments)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--native-root', type=Path, required=True)
    native_root = parser.parse_args().native_root
    env = RoboDojoWrapper(RoboDojoEnv(native_root))
    try:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation), 'reset observation outside observation_space'
        home = observation['arms']['left']['pose'].tolist()
        raised = [*home[:2], home[2] + .01, *home[3:]]
        move = lambda pose, **gripper: call('move', targets=[{'arm': 'left', 'pose': pose, **gripper}], profile='precision')
        script = [  # (tool call, expected native steps, expected `targets_reached`)
            (call('capture_scene'), 0, None),
            (call('hold', steps=2), 2, None),
            (move(raised), None, True),
            (move(raised, gripper=0.), None, True),
            (move(raised, gripper=1.), None, True),
            (move(home), None, True),
            (call('finish', status='success', summary='Native control check.'), 0, None),
        ]
        for action, steps, reached in script:
            before = observation['native_steps']
            observation, reward, terminated, truncated, info = env.step(action)
            feedback = json.loads(observation['feedback'])
            print(action['name'], feedback, 'gripper=%.3f' % observation['arms']['left']['gripper'], flush=True)
            assert env.observation_space.contains(observation), 'observation outside observation_space'
            assert feedback['ok'], feedback
            assert observation['native_steps'] - before == feedback['steps'], 'step count mismatch'
            assert steps is None or feedback['steps'] == steps, feedback
            assert feedback.get('targets_reached') == reached, feedback
            assert terminated == (action['name'] == 'finish') and not truncated, (terminated, truncated)
        print('verdict', env.unwrapped.evaluate())
        print('PASS: capture, hold, lift, gripper close/open, return, finish', flush=True)
    finally:
        env.close()
