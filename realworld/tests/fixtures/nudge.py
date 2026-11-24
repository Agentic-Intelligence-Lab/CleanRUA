"""Move each configured tool centre +5 mm in its own base frame, preserving its end effector."""
def main(robot):
    snapshot = robot.observe()
    measured_dx = {}
    for side, state in snapshot["arms"].items():
        target = state["pose"].copy()
        target[0] += 0.005
        robot.move(target, side=side)
        measured_dx[side] = robot.state(side=side)["pose"][0] - state["pose"][0]
    return {"views": sorted(snapshot["images"]), "measured_dx_m": measured_dx}


def run(robot, task, run_dir, *, max_decisions):
    """Standalone deterministic algorithm plug-in using the same policy as file replay."""
    import json

    if max_decisions < 1:
        return "max_decisions"
    result = main(robot)
    (run_dir / "measurement.json").write_text(json.dumps({"task": task, **result}, indent=2) + "\n")
    return "finish"
