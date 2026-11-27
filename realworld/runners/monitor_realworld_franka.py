"""Run the upstream Monitor method on a Franka workstation."""
from cleanrua.monitor import ACTIONS_PER_CHUNK, monitor
from ._station import load_policy, run_cli


def run(robot, instruction, args, out):
    reason = monitor(robot, load_policy(args.policy_config, ACTIONS_PER_CHUNK), instruction, max_decisions=args.max_decisions)
    return {"stop_reason": reason, "task_success": None}


def main(argv=None):
    run_cli("monitor", run, argv)


if __name__ == "__main__":
    main()
