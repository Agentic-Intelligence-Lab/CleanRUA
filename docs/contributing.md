# Contributing

## Adding an algorithm

CleanRUA follows CleanRL: **one file implements one algorithm**, and everything the algorithm needs is in that
file.

1. **Decide whether it is a new algorithm.** A change to what the model sends each turn, or to the interaction
   loop, is a new algorithm and gets a new file. A numeric setting or an on/off switch is a hyperparameter in
   `Args`, not a new file. Do not add several tricks to one algorithm.
2. **Name the file** `<paradigm>_<environment>.py`, for example `cap_robodojo.py`. A variant of a paradigm adds
   a word, for example `direct_memory_robodojo.py`.
3. **Copy, do not import.** Start from the closest existing file and change what differs. Algorithm files do not
   import from each other, so each can be read on its own.
4. **Keep the environment generic.** Tools are `@Tool` methods on the agent's `gym.Wrapper`. The environment in
   `cleanrua_utils/` contains only simulation, with no tools, prompts or models.
5. **Cite the source.** The first lines of the file name the method it follows.
6. **Test it.** Add CPU tests in `tests/` using `FakeRoboDojo` from `tests/test_direct_robodojo.py`, then run
   a short GPU check on a task that already succeeds with another agent.

```sh
python -m unittest discover -s tests -q                            # CPU: no simulator or model
python tests/smoke_robodojo.py --native-root /path/to/RoboDojo     # GPU: native control, no model
```

## Building this site

```sh
pip install -r docs/requirements.txt
mkdocs serve        # http://127.0.0.1:8000
mkdocs build --strict
```

## Acknowledgements

CleanRUA is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl),
[RoboSimStudio](https://github.com/robosimstudio/robosimstudio) and
[uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent). It uses
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) for simulation and
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) for action models. We thank their authors.
