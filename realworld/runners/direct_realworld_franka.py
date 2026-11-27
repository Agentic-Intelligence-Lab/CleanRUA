"""Run the upstream Direct method on a Franka workstation."""
from cleanrua.direct import direct
from ._station import run_cli


def run(robot, instruction, args, out):
    reason = direct(robot, instruction, max_decisions=args.max_decisions)
    return {"stop_reason": reason, "task_success": None}


def main(argv=None):
    run_cli("direct", run, argv)


if __name__ == "__main__":
    main()
