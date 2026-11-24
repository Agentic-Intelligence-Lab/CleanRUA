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
| 每侧可选的 `tool` | 外接末端填写自己的 `driver` 和 `args`。灵巧手还需在臂配置中提供 `hand_limits_rad: [下界数组, 上界数组]`，顺序与驱动一致。 |
| `cameras` | 以视角名为键，每路填写 `driver` 和 `args`。名称可用英文字母、数字、下划线和连字符；观测、发给模型的图像和保存的 PNG 均使用同一名称。 |

臂驱动的 `state()` 返回包含 `pose` 的字典。外接夹爪的 `state()` 返回 `{"gripper": 实测开度}`；一体式夹爪由臂状态提供该字段。运动方法与相机返回格式见下方[设备插件接口](#设备插件接口)。在主程序的 Python 环境中安装所用驱动和厂家 SDK。

仓库内置 `realworld.drivers.franky:FrankaArm`（`args.host` 为控制柜地址，使用独立末端时设置 `with_gripper=false`）和 `realworld.drivers.opencv:USBCamera`（`args.device` 为本机视频设备）。分别需要安装与硬件匹配的 `franky` 和提供 `cv2` 的 OpenCV 包。其他机械臂、夹爪和相机可按同一接口接入。

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

运行 `python -m realworld` 进入真机菜单，按提示选择算法、工作站和本轮自然语言任务。回车选择 Direct 和模拟设备；指定设备配置后须显式启用运动。Delegate 和 Monitor 可使用工作站内的动作策略或单独指定策略 JSON。

菜单和命令行默认使用 `api`，可通过 `--backend`、菜单或 `CLEANRUA_DECISION_BACKEND` 显式选择其他后端。API、local 和 Codex 的默认模型分别读取 `CLEANRUA_MODEL_NAME`、`CLEANRUA_LOCAL_MODEL` 和 `CLEANRUA_CODEX_MODEL`，Codex 未指定模型时由 CLI 选择。API 和 local 需提供服务所部署的模型 ID。思考强度读取 `CLEANRUA_REASONING_EFFORT`，缺省为 `default`，由模型服务决定。显式传入 `--reasoning-effort default` 也可恢复服务默认。支持的强度值取决于模型。

脚本可直接指定方法和参数，例如：

```sh
python -m realworld direct --task "检查桌面并结束" --backend codex --reasoning-effort medium
```

四个独立真机模块也接受 `--backend`、`--model`、`--reasoning-effort` 和 `--base-url`。API 路径可接第三方兼容端点，需要支持 PNG 图像、强制工具调用及多轮工具结果。local 默认连接 `http://127.0.0.1:8000/v1`，可设置 `CLEANRUA_LOCAL_BASE_URL` 和可选的 `CLEANRUA_LOCAL_API_KEY`。结果文件记录任务、决策后端、请求的模型和思考强度。模型调用 `finish` 时记录 `agent_finished=true`，`task_success` 留给使用者的任务评估逻辑。

四个可执行模块分别封装自己的工具、提示词和模型交互循环。四种方法使用相同的工作站配置和独立成功判据。设置 API 地址和模型名后，API 后端会优先读取 `CLEANRUA_API_KEY`；若未设置，会根据 HTTPS 端点从当前进程环境读取 OpenAI 的 `OPENAI_API_KEY`、阿里云 DashScope 的 `DASHSCOPE_API_KEY` 或 OpenRouter 的 `OPENROUTER_API_KEY`。自定义网关使用为它设置的 `CLEANRUA_API_KEY`。随后可用模拟设备运行下列命令。CaP 的隔离程序执行器需要 Linux、bubblewrap 和可导入 NumPy 的系统 Python。

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

安装与服务匹配的 `openpi_client`，将 driver 设为 `realworld.real_policy:DexJoCoOpenPI`。args 提供 `url`、`camera_mapping`（checkpoint 图像键到工作站相机名）、`joint_indices`（按 Allegro 原生顺序列出工作站手指索引，即 0 到 15 的一个排列）。客户端映射为 224×224 RGB、右臂优先的 46 维状态，单臂补零，将每侧 22 维 XYZ、旋转向量及手指目标转换成工作站对应臂的 pose 和 hand。该编码仅适用于匹配的右臂或双臂灵巧手配置，基座参考系、被控法兰和手指零位都必须与 checkpoint 对齐。物理双臂若使用各自的基座坐标，需在设备或策略适配器中完成与训练世界系的变换。普通 Franka 二指夹爪应选择匹配其训练数据的策略。

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

自定义算法使用可信模块的 `module:function` 入口。函数接收 `(bench, task, run_dir, *, max_decisions)`，返回结束原因。需要 VLA 的算法在签名中额外声明关键字参数 `policy`，启动器从策略配置注入 ChunkPolicy。仓库内的 `direct` 等名称继续按 `realworld.<方法>:run_<方法>` 解析。选择新算法和替换相机、臂、末端是相互独立的操作。

## 使用 ChatGPT 账号的 Codex CLI 做上层决策

在运行 Agent 的机器上安装 Codex CLI 并使用 ChatGPT 登录。`codex login status` 应显示 `Logged in using ChatGPT`。需要支持 `codex exec --ignore-user-config`、`--image`、`--output-schema` 和 `--output-last-message` 的版本。可用模拟工作站做一次只读决策：

```sh
CLEANRUA_DECISION_BACKEND=codex python -m realworld.direct \
  --task "Inspect the image and call finish" --max-decisions 1 --output-dir runs
```

适配器会检查登录方式，每轮将各相机的 PNG、状态、历史反馈及当前工具声明发给 CLI，通过结构化输出取得一条工具调用，再由环境校验并执行。CLI 失败会结束本次试验。默认后端为 `api`。ChatGPT 计划的额度、超额使用和信用点取决于账号设置，参见 [OpenAI 的 Codex 计划说明](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)。

四个 `*_real` 入口均支持该选项。Delegate 和 Monitor 的 VLA 仍通过自己的策略客户端运行。其他 CLI 可通过 `CLEANRUA_DECISION_BACKEND=module:factory` 提供适配器：factory 接收当前工具列表，返回 `decide(messages)`，其结果为含一条 tool_calls 的 assistant 消息。各供应商的认证、图片输入及输出解析由对应适配器负责。
