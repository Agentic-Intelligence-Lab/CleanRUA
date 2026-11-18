# Overview

CleanRUA (Clean Implementations of Robot-Use Agents) provides single-file implementations of robot-use agents
for research, following [CleanRL](https://github.com/vwxyzjn/cleanrl). Each file contains a whole agent: its
prompt, its tools and the interaction loop. The environments are plain [Gymnasium](https://gymnasium.farama.org/)
environments.

![Head-camera views of three successful RoboDojo episodes: delegate stacking bowls, direct stacking blocks, and direct lifting scissors](demo.gif){ width="100%" }

- **Single file.** One algorithm per file, readable top to bottom.
- **Plain environments.** Tools and models live in the agent, not in the environment.
- **Official verdicts.** Success always comes from the environment's own check.

Start with the [installation guide](get-started/installation.md), then read the
[algorithms overview](algorithms/overview.md).

## Citing CleanRUA

```bibtex
@misc{cleanrua2026yuan,
  title        = {CleanRUA: Clean Implementations of Robot-Use Agents},
  author       = {Mingqi Yuan, Xudong Wu, Jiayu Chen},
  year         = {2026},
  howpublished = {\url{https://github.com/Agentic-Intelligence-Lab/CleanRUA}},
}
```
