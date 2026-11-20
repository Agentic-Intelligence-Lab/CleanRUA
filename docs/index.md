# Overview

CleanRUA (Clean Implementations of Robot-Use Agents) provides single-file implementations of robot-use agents
for research, following [CleanRL](https://github.com/vwxyzjn/cleanrl). Each file contains a whole agent: its
prompt, its tools and the interaction loop. The environments are plain [Gymnasium](https://gymnasium.farama.org/)
environments.

![Successful episodes. Top, RoboDojo head camera: delegate stacking bowls, direct stacking blocks and direct lifting scissors. Bottom, DexJoCo ego camera: delegate clicking the mouse on its pad, delegate hammering the nail and monitor lifting the bucket with the boxed food inside](demo.gif){ width="100%" }

- **Single file.** One algorithm per file, readable top to bottom.
- **Plain environments.** Tools and models live in the agent, not in the environment.
- **Official verdicts.** Success always comes from the environment's own check.

Start with the installation guide for [RoboDojo](get-started/installation_robodojo.md) or
[DexJoCo](get-started/installation_dexjoco.md), then read the
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
