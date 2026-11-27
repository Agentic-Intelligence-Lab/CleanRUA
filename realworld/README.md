# CleanRUA 可配置真机环境

在能访问设备的 Linux 工作站上准备 Python 3.10 或更高版本。克隆后从仓库根目录运行下列 Bash 命令。设备驱动、标定和工作站 JSON 由使用者管理。首次接入按以下顺序进行。

## 1. 获取分支并安装

```bash
git clone --branch Realworld_Dev --single-branch https://github.com/Agentic-Intelligence-Lab/CleanRUA.git
cd CleanRUA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy pillow httpx "gymnasium==1.2.3" "pydantic>=2.11,<3" "tyro>=0.9,<1"
```

主程序使用虚拟环境。CaP 会通过 Linux 的 `bwrap` 启动隔离的 `/usr/bin/python3` 来执行模型写的代码。Debian/Ubuntu 上缺少依赖时可安装 `python3-venv bubblewrap python3-numpy`。程序若使用 OpenCV，也要在执行它的 Python 环境中安装。以下命令可在接设备前检查基础环境：

```bash
bwrap --version
/usr/bin/python3 -c 'import numpy'
python -m unittest discover -s realworld/tests -v
python -m realworld check --out "runs/offline-$(date +%s)"
```

## 2. 接入自己的机械臂和相机

创建一个仓库外的目录存放 `workstation.json` 和自定义驱动。每个设备用 `driver` 指定可导入的 `模块名:类名`，用 `args` 提供该类的构造参数。若驱动以本地 Python 文件提供，将它所在的目录加入 `PYTHONPATH`；已安装到虚拟环境的驱动包可直接导入。仅将 JSON 与驱动放在同一目录不足以让 Python 找到驱动。

| JSON 字段 | 要填写的内容 |
|---|---|
| `arms.left`、`arms.right` | 按实际臂数保留相应的键。每侧填写 `driver`、`args` 和 `end_effector`（`gripper`、`hand` 或 `none`）。 |
| 每侧的 `workspace_m` | 两个三元素数组 `[[xmin, ymin, zmin], [xmax, ymax, zmax]]`，采用该臂基座系和现场可用范围，单位米。 |
| 每侧的 `max_translation_m`、`max_rotation_rad` | 单步允许的三维位移长度和相对旋转角，分别以米、弧度表示，均为正数。 |
| 每侧可选的 `arrival_translation_m`、`arrival_rotation_rad` | 运动完成后，实测 TCP 与目标之间允许的位置和姿态误差。缺省分别为 0.002 m 和 0.05 rad；单步上限和设备控制器保护独立生效。 |
| 每侧可选的 `tool` | 外接末端填写自己的 `driver` 和 `args`。灵巧手还需在臂配置中提供 `hand_limits_rad: [下界数组, 上界数组]`，顺序与驱动一致。 |
| `cameras` | 以视角名为键，每路填写 `driver` 和 `args`。名称可用英文字母、数字、下划线和连字符；观测、发给模型的图像和保存的 PNG 均使用同一名称。 |

臂驱动的 `state()` 返回包含 `pose` 的字典。外接夹爪的 `state()` 返回 `{"gripper": 实测开度}`；一体式夹爪由臂状态提供该字段。运动方法与相机返回格式见下方[设备插件接口](#设备插件接口)。在主程序的 Python 环境中安装所用驱动和厂家 SDK。

设备适配器随工作站维护。升级已有配置时，先将适配器放入现场 Python 包，将 `arms.*.driver`、`arms.*.tool.driver` 和 `cameras.*.driver` 指向该包，保留已经标定的 `args` 和动作范围。早期只含 `host`、`camera` 的配置也需要补全这两个具名设备表。先比较新旧适配器的只读观测，再启用运动。

## 3. 只读检查工作站

把第一行替换为自己的目录。在同一个终端继续操作：

```bash
STATION_DIR=/absolute/path/to/your/station
STATION="$STATION_DIR/workstation.json"
export PYTHONPATH="$STATION_DIR${PYTHONPATH:+:$PYTHONPATH}"
CHECK_DIR="runs/check-$(date +%s)"
python -m realworld check --config "$STATION" --out "$CHECK_DIR"
```

核对 `$CHECK_DIR/result.json` 中 `result.arms` 的实测位姿和末端状态，并打开每个 `<视角名>.png` 核对内容、颜色和相机名称。此检查无需模型密钥；驱动构造和读取应只连接设备并返回状态。省略 `--config` 会使用内置模拟设备。每次 `--out` 请使用新目录。

### 遥操采点并写入工作区

初次标定可以先省略待测臂的 `workspace_m`。启动自己的遥操或示教工具，在另一个配置好驱动路径的终端运行：

```bash
python -m realworld calibrate-workspace --config "$STATION" --arm left
```

单臂可省略 `--arm`；双臂每次明确选择 `left` 或 `right`。采点程序只打开所选臂的位置读取接口，无需相机、模型或 `--enable-motion`。如果遥操已独占 SDK 连接，可在该臂的 JSON 下添加可选的 `state_reader`，用 `driver` 和 `args` 指向订阅遥操反馈的现场驱动。它的 `state()` 应返回同一个基座系、同一个 TCP 的实测六维 `pose`，构造和关闭均不应接管机械臂。

1. 用遥操将 TCP 移到任务所需边界，停稳后按一次 **回车**。终端显示点号和基座系 XYZ（米）。依次覆盖 X、Y、Z 的两端。
2. 输入 `done`。程序对所有采样点逐轴取最小值、最大值，显示旧值和新值 `[[xmin, ymin, zmin], [xmax, ymax, zmax]]`。任一轴跨度为零时可以继续采点。
3. 确认这个长方体适合当前工具、桌面和夹具后输入 `y`，写回所选臂的 `workspace_m`。采点中输入 `q`、按 Ctrl-C，或在确认时直接回车，都会保留原配置。

这个边界框用于约束 TCP 的当前位置和目标位置；采点本身无法验证框内所有姿态可达或整臂路径无碰撞。再次启动实验时会加载新边界。两条臂可分别标定。

## 4. 用 API 运行 CaP 并生成控制器

准备一个支持图像输入和工具调用的模型。输入供应商的 API 基地址（含 `/v1`）、模型 ID、本局任务和密钥。下面的 `--enable-motion` 会允许模型通过上述工作站配置指挥设备，请在现场已可执行任务时运行。

本分支的兼容端点走 Chat Completions；使用 OpenAI 官方基地址 `https://api.openai.com/v1` 和 `gpt-6.2-astra` 时走 Responses。若供应商或模型仅支持其中一种协议，需选择与此入口匹配的组合。

```bash
read -rp 'API base URL: ' CLEANRUA_BASE_URL
read -rp 'Model ID: ' CLEANRUA_MODEL_NAME
read -rsp 'API key: ' CLEANRUA_API_KEY; echo
export CLEANRUA_BASE_URL CLEANRUA_MODEL_NAME CLEANRUA_API_KEY
read -rp 'Task: ' TASK
RUN_DIR="runs/cap-$(date +%s)"
python -m realworld.cap --backend api --config "$STATION" --enable-motion \
  --task "$TASK" --out "$RUN_DIR"
```

`trajectory.jsonl` 保存每轮的观测引用、生成代码和工具反馈。探索正常结束后会再请求同一 API，生成 `$RUN_DIR/controller.py`，入口为 `main(robot)`。在 `result.json` 中核对 `controller` 字段。`controller_verified=false` 表示已保存通过语法和入口检查的候选策略，跨局效果需要独立评估。

## 5. 补生成和跨局回放

若生成请求失败，保留日志和对应 PNG。确认 `RUN_DIR` 指向有结束或故障记录及观测的那一局，在上述 API 环境中运行：

```bash
python -m realworld.cap compile --backend api --run-dir "$RUN_DIR" --config "$STATION"
```

这是仅在缺少 `controller.py` 时使用的补生成步骤，无需连接机器人。已有策略时直接进入回放。审阅策略及其依赖、重新布置场景后执行：

```bash
python -m realworld replay --config "$STATION" --enable-motion \
  --policy "$RUN_DIR/controller.py" --out "runs/replay-$(date +%s)"
```

回放使用工作站 Python 执行受信任的策略文件，无需 API key。用新一局的实测结果检查是否完成任务。

## 策略接口

每个控制步由臂名和目标字段组成。例如 `{"left": {"pose": [0.40, 0, 0.30, 0, 0, 0], "gripper": 0.5}}`。pose 是基座系的绝对 XYZ（米）与固定轴 XYZ 欧拉角（弧度）；gripper 是 0 到 1 的开度；hand 是按驱动关节顺序排列的弧度数组。只提供需要更新的臂和字段即可。多臂命令先校验再顺序执行。策略插件负责 checkpoint 自身的坐标变换、归一化和原生输入输出维度。

## 运行算法

`--max-decisions` 默认最多 30 轮；正整数指定轮数，显式传 `0` 则持续决策，直到方法结束、环境终止或出现未处理异常。

运行 `python -m realworld` 进入真机菜单，按提示选择算法、工作站和本轮自然语言任务。回车选择 Direct 和模拟设备；指定设备配置后须显式启用运动。Delegate 和 Monitor 可使用工作站内的动作策略或单独指定策略 JSON。

菜单和命令行默认使用 `api`，可通过 `--backend`、菜单或 `CLEANRUA_DECISION_BACKEND` 显式选择其他后端。API、local 和 Codex 的默认模型分别读取 `CLEANRUA_MODEL_NAME`、`CLEANRUA_LOCAL_MODEL` 和 `CLEANRUA_CODEX_MODEL`，Codex 未指定模型时由 CLI 选择。API 和 local 需提供服务所部署的模型 ID。思考强度读取 `CLEANRUA_REASONING_EFFORT`，缺省为 `default`，由模型服务决定。显式传入 `--reasoning-effort default` 也可恢复服务默认。支持的强度值取决于模型。

脚本可直接指定方法和参数，例如：

```sh
python -m realworld direct --task "检查桌面并结束" --backend codex --reasoning-effort medium
```

四个独立真机模块也接受 `--backend`、`--model`、`--reasoning-effort` 和 `--base-url`。API 路径可接第三方兼容端点，需要支持 PNG 图像、强制工具调用及多轮工具结果。local 默认连接 `http://127.0.0.1:8000/v1`，可设置 `CLEANRUA_LOCAL_BASE_URL` 和可选的 `CLEANRUA_LOCAL_API_KEY`。结果文件记录任务、决策后端、请求的模型和思考强度。模型调用 `finish` 时记录 `agent_finished=true`，`task_success` 留给使用者的任务评估逻辑。

四个方法各自定义策略工具与提示词，共用模型后端、决策回合和记录格式。四种方法使用相同的工作站配置和独立成功判据。设置 API 地址和模型名后，API 后端会优先读取 `CLEANRUA_API_KEY`；若未设置，会根据 HTTPS 端点从当前进程环境读取 OpenAI 的 `OPENAI_API_KEY`、阿里云 DashScope 的 `DASHSCOPE_API_KEY` 或 OpenRouter 的 `OPENROUTER_API_KEY`。自定义网关使用为它设置的 `CLEANRUA_API_KEY`。随后可用模拟设备运行下列命令。CaP 的隔离程序执行器需要 Linux、bubblewrap 和可导入 NumPy 的系统 Python。

```sh
python -m realworld.direct --task "Read the pose and finish" --output-dir runs/direct
python -m realworld.cap --task "Read the pose and finish" --output-dir runs/cap
```

指定真实设备需传入 `--config`，并在已确认的现场实验窗口显式启用 `--enable-motion`。独立的 `realworld.__main__ check` 和 `replay` 命令用于只读检查与已审阅程序的文件回放。finish 结束回合，实物任务结果另行记录。

## Delegate 和 Monitor

Delegate 的 `actor_act(instruction, steps, active_groups=None)` 向动作策略发送当前阶段的指令。可指定参与的臂，未选中的臂不会收到写命令。Monitor 的 `run_policy(steps)` 始终传入启动时的任务指令。两个方法都提供 `move` 纠正动作和 `finish`，每次工具结束后大模型重新获得观测。

```sh
python -m realworld.delegate --task "Place the block in the tray" --policy-config "$POLICY_CONFIG" --output-dir runs
python -m realworld.monitor --task "Place the block in the tray" --policy-config "$POLICY_CONFIG" --output-dir runs
```

上述命令使用模拟机械臂，也需要实际可用的模型端点和策略服务。切换设备时另外传入工作站配置。策略客户端也可放在工作站 JSON 的 `policy` 字段，单独的 `--policy-config` 优先。选择适合该 checkpoint 的动作前缀长度 `actions_per_chunk` 和控制频率 `control_hz`。每次重规划使用新的状态和相机帧，只执行预测前缀；每一行都经过环境动作检查。频率是发送命令的上限；设备端阻塞动作与采图耗时会降低实际频率。每个已派发动作及其测量回执写入 trajectory.jsonl。多臂仍顺序执行。

### 具名动作 HTTP 服务

在策略配置的 `args.url` 中填写自己的推理服务地址。请求包含 `contract`、`instruction`、按工作站名称索引的 `images_png`（RGB PNG 的 base64）、按臂名索引的实测 `arms`、各相机和状态的主机接收时间。响应应回传同一个 contract 字符串，并在 `actions` 返回按时间排列的非空具名命令列表，例如 `[{"left":{"gripper":0.5}}]`。contract 为 `arms.absolute.base_m.xyz_rpy_rad.open_fraction.hand_rad.v1`。服务端负责 checkpoint 的相机预处理、归一化以及增量到绝对目标的转换。无状态预测服务可直接使用这一协议。

### DexJoCo OpenPI 原生服务

安装与服务匹配的 `openpi_client`，将 driver 设为 `realworld.policy:DexJoCoOpenPI`。args 提供 `url`、`camera_mapping`（checkpoint 图像键到工作站相机名）、`joint_indices`（按 Allegro 原生顺序列出工作站手指索引，即 0 到 15 的一个排列）。客户端映射为 224×224 RGB、右臂优先的 46 维状态，单臂补零，将每侧 22 维 XYZ、旋转向量及手指目标转换成工作站对应臂的 pose 和 hand。该编码仅适用于匹配的右臂或双臂灵巧手配置，基座参考系、被控法兰和手指零位都必须与 checkpoint 对齐。物理双臂若使用各自的基座坐标，需在设备或策略适配器中完成与训练世界系的变换。

其他 checkpoint 可通过可信配置中的 `module:Class` 提供客户端；实现 `predict(observation, instruction)`，返回上述物理单位的绝对动作序列即可。环境及两个 Agent 无需修改。

## 设备插件接口

| 插件 | 驱动需实现的接口 |
|---|---|
| 机械臂 | `state()` 提供六维基座系 XYZ/RPY，支持可选 joints；`move_pose(pose)` 阻塞到完成或报错；`stop()` 请求停止 |
| 夹爪或灵巧手 | 一体化设备或外接 tool 提供 `state()`、`set_gripper(opening)` 或 `set_hand(joints)`、`stop()`。开度为 0 到 1，手指关节按配置顺序使用弧度 |
| RGB 相机 | `read()` 返回 H×W×3 的 uint8 RGB，`close()` 释放资源 |
| 动作策略 | `predict(observation, instruction)` 返回具名的绝对动作序列 |

`robot.observe()` 在 CaP 程序与受信任文件回放中返回相同结构：按名称索引的 arms 与 images 和接收时间戳。单臂和双臂都用 `obs["arms"][side]["pose"]`；只读某只臂可用 `robot.state(side=side)`。无参 state() 保留旧策略的数据格式。设备可返回 NumPy 数组或标量，公开的具名臂数据使用普通列表和数值。隔离的 CaP 程序可读取完整 RGB 像素，程序代码只通过父进程请求设备操作。

设备执行异常时，本次命令选中的臂收到 stop，本次会话随后拒绝运动。工作站级硬件联停由现场控制系统负责。

CaP 单个代码块的等待预算为 600 秒，隔离进程虚拟地址空间上限为 2 GiB。若运动异常已触发环境的运动锁，代码块回执会携带 `motion_fault`、`motion_enabled=false` 和 `recovery_required=true`。回执包含当时请求的动作及可读的实测状态；状态读取失败时记录读取错误。当前会话后续的移动请求会被拒绝。

自定义算法使用可信模块的 `module:function` 入口。函数接收 `(bench, task, run_dir, *, max_decisions)`，返回结束原因。需要 VLA 的算法在签名中额外声明关键字参数 `policy`，启动器从策略配置注入 ChunkPolicy。仓库内的 `direct` 等名称继续按 `realworld.<方法>:run_<方法>` 解析。选择新算法和替换相机、臂、末端是相互独立的操作。

## 使用 ChatGPT 账号的 Codex CLI 做上层决策

在运行 Agent 的机器上安装 Codex CLI 并使用 ChatGPT 登录。`codex login status` 应显示 `Logged in using ChatGPT`。需要支持 `codex exec --ignore-user-config`、`--image`、`--output-schema` 和 `--output-last-message` 的版本。可用模拟工作站做一次只读决策：

```sh
CLEANRUA_DECISION_BACKEND=codex python -m realworld.direct \
  --task "Inspect the image and call finish" --max-decisions 1 --output-dir runs
```

适配器会检查登录方式，每轮将各相机的 PNG、状态、历史反馈及当前工具声明发给 CLI，通过结构化输出取得一条工具调用，再由环境校验并执行。CLI 失败会结束本次试验。默认后端为 `api`。ChatGPT 计划的额度、超额使用和信用点取决于账号设置，参见 [OpenAI 的 Codex 计划说明](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)。

四种方法均支持该选项。Delegate 和 Monitor 的 VLA 仍通过自己的策略客户端运行。其他 CLI 可通过 `CLEANRUA_DECISION_BACKEND=module:factory` 提供适配器：factory 接收当前工具列表，返回 `decide(messages)`，其结果为含一条 tool_calls 的 assistant 消息。各供应商的认证、图片输入及输出解析由对应适配器负责。


## 代码结构

| 文件 | 职责 |
|---|---|
| `direct.py`、`cap.py`、`delegate.py`、`monitor.py` | 复用各上游方法的提示词，提供工作站专属工具。CaP 在探索完成后生成控制器。 |
| `agent.py`、`episode.py` | 校验并执行工具调用；组装每轮模型输入，记录动作、实测回执、终止原因和故障。仿真原型和真机共用 `cleanrua_utils.agent` 中的图像编码及工具决策函数。 |
| `model.py`、`policy.py` | 分别连接决策模型和动作策略。 |
| `program.py` | 在隔离 Python 会话中执行 CaP 程序，经父进程访问工作站。 |
| `workstation.py`、`env.py` | 按 JSON 导入现场驱动，提供统一的观测、运动校验、执行与回读。 |
| `calibration.py` | 遥操时读取所选臂的 TCP，计算并按确认结果保存工作区。 |
| `__main__.py` | 四种方法的命令行入口、只读检查和受信任策略回放。 |

## 只读获取工作站信息

从展开后的 URDF 提取关节名称、类型、运动链、关节限制、mimic 关系、固定变换和根/末端候选链接：

```bash
python3 -m realworld.inspect --urdf /path/to/robot.urdf --out robot-inventory.json
```

在已安装 ROS 2 的工作站上，使用能够导入 rclpy 的 Python。先加载现场 ROS 环境，并采用设备实际使用的 ROS_DOMAIN_ID，再运行：

```bash
python3 -m realworld.inspect --ros --seconds 5 --out station-inventory.json
```

工具在指定时间内发现话题，接收 JointState、Image、CameraInfo 和 TF 的元数据，并通过 GetParameters 查询 robot_description，通过 ListControllers 查询控制器名称、状态和接口。robot_description 话题同样可作为模型来源。图像报告保存分辨率、编码、帧名称和时间戳；像素留在传输层。URDF 离线解析只依赖 Python 标准库，ROS 模式依赖现场安装的消息和服务包。

报告保留源文件、ROS 话题/服务名和接收时间。没有收到数据的话题只会出现在 topics，查询失败会出现在 errors。confirmation_required 列出仍需确认的任务运动链、TCP、执行控制器、现场标定、障碍物和工作区。该报告用于填写工作站配置。当前工具提供读取和清点功能，厂家 CAN/SDK 设备需先由相应驱动暴露信息。
