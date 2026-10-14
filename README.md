# 本地 GRPO / DAPO 多步工具调用 Agent

这是一个离线优先的强化学习研究工程。项目使用合成订单处理任务，训练本地开源语言模型完成多步工具调用，并以标准 GRPO 为基线研究 DAPO 机制。

订单环境、任务生成、工具执行、奖励计算和 CPU 单元测试不依赖外部大模型 API。真实模型推理与训练使用 Transformers、TRL 和 PEFT。

## 当前状态

已实现并经过 CPU 测试的部分：

- 确定性合成任务和独立的训练、验证、测试文件。
- `query_order`、`query_stock`、`modify_order`、`verify_order` 四个工具。
- 严格 JSON 工具协议、多轮 rollout、环境状态验证和完整轨迹。
- 基于真实环境状态的六分项奖励；模型仅声称“成功”不能获得成功奖励。
- Oracle smoke policy、基础评估指标和标准 TRL `GRPOTrainer` 入口。
- Oracle 轨迹冷启动数据构造：同一条轨迹的 plan 视图与单步视图。
- 格式部分信用与 markdown 代码围栏诊断；只参与打分，不改变执行的严格解析。
- 含 `adapter_config.json` 的目录自动按 LoRA adapter 加载。
- Clip-Higher、token-level loss、零方差组过滤和 overlong shaping 的纯 PyTorch 数值参考。
- 独立的最小 GRPO 参考公式及反向传播测试。

已在 GPU 环境实际运行通过的部分：

- 真实 Qwen2.5-0.5B-Instruct 推理、冷启动 SFT、GRPO 参数更新和训练后评估。
- LoRA adapter 保存与重新加载（`model_loader` 自动识别 adapter 目录）。
- 格式部分信用与 few-shot 协议示例，用于把组内奖励方差从 0 变为非 0。

尚未完成的部分：

- 当前 `dynamic_sampling()` 是零方差组过滤原语，不是带重采样预算的完整 Dynamic Sampling 训练循环。
- 尚未完成六组正式消融、三随机种子实验和 DAPO 相对 Vanilla GRPO 的研究结论。
- 现有 GRPO 结果只有 10 个优化步，属于冒烟规模，不足以判断算法有效性。

## 实测记录

环境：NVIDIA A10（23 GB）、`torch 2.10.0+cu128`、`transformers 5.1.0`、`trl 1.15.0`、
`peft 0.21.2`、`datasets 5.1.0`、`accelerate 1.15.0`。配置为 `configs/baseline.yaml`，
固定种子 20261013，测试集 32 条。

| 阶段 | task_success_rate | average_reward | average_tool_calls | invalid_tool_call_rate |
|---|---|---|---|---|
| 基座模型（零样本） | 0 | -1.0 | 0.0 | — |
| 冷启动 SFT 后 | 0.625 | 1.216 | 2.438 | 0.282 |
| GRPO 10 步后 | 0.625 | 1.292 | 2.656 | 0.235 |

冷启动 SFT：276 条样本（64 条 plan 视图 + 212 条单步视图），3 个 epoch，`train_loss` 0.2086。
GRPO：`train_loss` 0.3173，`reward_std` 最高 2.125（不再是全组 0），
`parameter_update_verified: true`，336 个可训练张量，`global_step` 10。

两点必须说明：基座模型的失败原因是输出被 markdown 代码围栏包裹且字段名不符合协议，
所有候选奖励因此恒等于 `invalid_action`；GRPO 阶段成功率与冷启动持平，只有奖励和无效调用率
小幅改善，10 步不足以支撑任何算法结论。这些数字仅证明链路可跑通。

口径说明：冷启动与 GRPO 两行为全量 32 条测试集；基座模型一行为 5 条样本，其全部解析失败、
零工具调用，扩大样本量不会改变该结论。

详细审计见 [实现审计](docs/implementation_audit.md)。任何未运行实验都不得填写虚构数字。

## 项目结构

```text
grpo_agent/             订单环境、Agent、奖励、格式信用、DAPO 数值核心
training/               TRL GRPO 训练入口、冷启动 SFT 和扩展接口
evaluation/             固定测试集评估与指标
scripts/                数据生成和 CPU smoke test
configs/                模型、训练、奖励和路径配置
tests/                  不依赖 GPU/大模型的单元测试
data/generated/         生成的数据集
logs/                   轨迹和评估输出
checkpoints/            模型或 LoRA adapter 输出
```

## 推荐模型与加载方式

默认模型为 `Qwen/Qwen2.5-0.5B-Instruct`。模型来源按以下优先级解析：

1. 环境变量 `MODEL_PATH`；
2. `configs/model.yaml` 中的 `model.local_path`；
3. 环境变量 `MODEL_ID`；
4. YAML 中的 `model.name`。

默认禁止联网下载。下载失败不会回退到商业模型 API。

## 安装

建议创建独立虚拟环境，不要修改已有科研环境：

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux / 百度 AI Studio：

```bash
source .venv/bin/activate
python -m pip install -r requirements-aistudio.txt
```

`requirements-aistudio.txt` 不安装 PyTorch，以免替换平台已经匹配 CUDA 的版本。依赖文件当前给出的是兼容范围，不是已经在目标 GPU 环境验证过的锁文件。安装前先执行：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

不要在未确认 CUDA 兼容性时盲目替换平台的 PyTorch。应在目标 AI Studio 镜像验证 TRL API 后，再保存精确依赖快照。

在 A10 / CUDA 12.8 镜像上实际验证通过的组合：`torch 2.10.0+cu128`、`transformers 5.1.0`、
`trl 1.15.0`、`peft 0.21.2`、`datasets 5.1.0`、`accelerate 1.15.0`。
`requirements-aistudio.txt` 里的 `transformers<5` 与 `trl<0.30` 上界对应的是旧版本组合，
新环境下以"不覆盖平台 PyTorch"为前提按 pip 实际解析结果安装即可，装完用下面的命令确认版本。

TRL 1.15 的 `GRPOConfig` 已移除 `max_prompt_length`。`training/train_grpo.py` 会先探测该字段是否
存在：存在则原样传入，不存在则改用 tokenizer 在构建数据集前截断，并把实际生效方式写进
`run_manifest.json` 的 `prompt_length_control`，不会静默丢弃配置。`--dry-run` 会打印当前生效的模式。

## 配置与云端路径覆盖

业务参数保存在 YAML 中；机器相关路径可通过环境变量覆盖。因此上传云端后不需要修改 Python 文件：

| 环境变量 | 用途 |
|---|---|
| `MODEL_PATH` | 本地或云端挂载的完整模型目录 |
| `MODEL_ID` | 没有本地模型时使用的 Hugging Face 仓库 ID |
| `TOKENIZER_PATH` | 可选的独立 tokenizer 目录 |
| `ALLOW_MODEL_DOWNLOAD` | `1/true` 允许首次联网下载，默认禁止 |
| `GRPO_OUTPUT_ROOT` | checkpoint、轨迹和评估结果的持久化根目录 |

百度 AI Studio 示例：

```bash
export MODEL_PATH=/root/paddlejob/workspace/models/Qwen2.5-0.5B-Instruct
export GRPO_OUTPUT_ROOT=/root/paddlejob/workspace/output/grpo-agent
export ALLOW_MODEL_DOWNLOAD=0
```

如果平台持久化目录不同，只需修改环境变量。相对数据路径默认以项目根目录为工作目录。

## 端到端运行顺序

以下命令均在项目根目录执行。首次运行时建议保留 `GRPO_OUTPUT_ROOT` 之外的所有默认值。

```bash
# 0) 确认 GPU 与 CUDA 版 PyTorch 没被替换
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 1) 生成固定数据集，跑 CPU 闭环（不需要 GPU、不需要模型权重）
python scripts/generate_data.py --config configs/baseline.yaml
python -m unittest discover -s tests
python scripts/smoke_test.py --config configs/baseline.yaml

# 2) 冷启动 SFT：用 Oracle 轨迹教会模型工具协议（详见下一节）
python training/sft_coldstart.py --config configs/baseline.yaml --dry-run
python training/sft_coldstart.py --config configs/baseline.yaml
# 记下它打印的输出目录，例如 checkpoints/sft_coldstart

# 3) 以冷启动模型为起点做 GRPO
export MODEL_PATH=checkpoints/sft_coldstart
python training/train_grpo.py --config configs/baseline.yaml --dry-run
python training/train_grpo.py --config configs/baseline.yaml
# 产物目录见 configs/baseline.yaml 的 training.output_dir，例如 checkpoints/baseline

# 4) 训练后评估（MODEL_PATH 指向第 3 步的 adapter 目录）
export MODEL_PATH=checkpoints/baseline
python evaluation/evaluate.py --config configs/baseline.yaml --policy model
```

只想验证管线而不训练，可以跳过 3、4，用 Oracle 策略跑一次评估：

```bash
python evaluation/evaluate.py --config configs/baseline.yaml --policy oracle
```

每一步都会做硬校验：数据为空、Oracle 轨迹失败、参数没有真实变化、global step 小于 1
都会直接抛错退出，不会产出看似成功的结果。

## CPU 最小闭环

在项目根目录执行：

```bash
python scripts/generate_data.py --config configs/baseline.yaml
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/baseline.yaml
python evaluation/evaluate.py --config configs/baseline.yaml --policy oracle
```

这些命令验证环境、工具、奖励和评估管线，不代表真实模型训练完成。

## 本地模型评估

```bash
python evaluation/evaluate.py --config configs/baseline.yaml --policy model --limit 20
```

模型每轮必须输出一个 JSON 对象，例如：

```json
{"tool":"query_order","arguments":{"order_id":"ORD-000001"}}
```

或：

```json
{"final":"最终说明"}
```

非法 JSON、未知工具、错误参数和超过交互上限都会被记录，不会被静默修复。

## 冷启动 SFT

`Qwen2.5-0.5B-Instruct` 在零样本下几乎不会输出合法工具协议，组内奖励恒等于
`invalid_action`，GRPO 因零方差而没有任何优势信号，训练脚本会主动以
`Trainer ran but no trainable parameter value changed` 失败。先用确定性 Oracle
轨迹做一轮监督微调，把协议格式教会模型，再进入 GRPO：

```bash
python training/sft_coldstart.py --config configs/baseline.yaml --dry-run
python training/sft_coldstart.py --config configs/baseline.yaml
```

脚本会在隐藏环境中回放 Oracle，并把同一条轨迹序列化成两种视图：完整 plan 数组
（GRPO 优化的对象）与逐步单条动作（评估器要求的形式），避免训练与评估的模态错配。
只有 Oracle 成功的轨迹才会进入训练集，参数未变化时同样会失败退出。

训练结束后用该目录作为 GRPO 的起点：

```bash
export MODEL_PATH=<上一步输出目录>
python training/train_grpo.py --config configs/baseline.yaml
```

`grpo_agent/format_reward.py` 提供格式部分信用：按结构合法率、工具名合法率、参数
合法率给出连续分值，并对 markdown 代码围栏扣分。它只参与打分，不改变执行的严格解析，
因此"围栏但结构正确"的候选会高于"完全乱写"的候选，从而打破组内零方差，同时梯度方向
仍然指向"去掉围栏"。权重位于 `configs/baseline.yaml` 的 `reward.format` 段，可整体关闭以做消融。

## 标准 GRPO 训练

先检查数据、奖励回放和当前 TRL 构造接口：

```bash
python training/train_grpo.py --config configs/baseline.yaml --dry-run
```

再运行真实训练：

```bash
python training/train_grpo.py --config configs/baseline.yaml
```

训练入口会加载固定任务、生成每组多个 completion、在全新隐藏环境中回放工具计划、根据实际状态计算奖励，并由 TRL 执行反向传播。训练结束后还会检查可训练参数是否真实变化，并保存模型或 LoRA adapter、tokenizer 和 `run_manifest.json`。

当前标准 TRL 路径优化“完整生成的工具计划”。它不是在一次 generation 中逐 token 接收实时工具返回。独立 rollout 引擎支持推理和评估时的真实多轮反馈，二者边界不能混淆。

## DAPO 接入原则

目标 TRL 版本若原生支持，应优先使用：

- `epsilon` 与 `epsilon_high`：Clip-Higher；
- `loss_type="dapo"`：token-level DAPO loss；
- completion 截断 mask：屏蔽不完整样本。

软 Overlong Reward Shaping 应由真实 token 长度驱动奖励函数。Dynamic Sampling 需要在生成后检查组内奖励方差，并在有效组不足时继续采样，直到满足 batch 或耗尽明确的重采样预算。

`grpo_agent/dapo.py` 是独立数值参考，不表示已经替换或分叉 TRL Trainer。接入前必须在目标环境检查接口：

```bash
python -c "import inspect; from trl import GRPOConfig, GRPOTrainer; print(inspect.signature(GRPOConfig.__init__)); print(inspect.signature(GRPOTrainer.__init__))"
```

## 产物存放位置

未设置 `GRPO_OUTPUT_ROOT` 时，一切写在项目目录内；设置后 checkpoint、轨迹和评估结果改写到
该根目录，数据集路径不受影响（仍由 YAML 控制）。

| 产物 | 默认位置 | 设置 `GRPO_OUTPUT_ROOT=<root>` 后 |
|---|---|---|
| 数据集 | `data/generated/{train,validation,test}.jsonl` | 不变 |
| 冷启动模型 | `checkpoints/sft_coldstart/` | `<root>/checkpoints/sft_coldstart/` |
| 冷启动清单 | 上述目录下的 `sft_manifest.json` | 同左 |
| GRPO 模型/adapter | `checkpoints/baseline/`（YAML 的 `training.output_dir`） | `<root>/checkpoints/standard_grpo_baseline/`（YAML 的 `experiment`） |
| 中间 checkpoint | 上述目录下的 `checkpoint-<step>/` | 同左 |
| GRPO 清单 | 上述目录下的 `run_manifest.json` | 同左 |
| 评估轨迹 | `logs/traces/evaluate_<policy>.jsonl` | `<root>/logs/traces/` |
| 评估指标 | `logs/evaluation/<policy>_metrics.json` | `<root>/logs/evaluation/` |

`run_manifest.json` 记录 `global_step`、`parameter_update_verified`、`trainable_parameter_tensors`、
`prompt_length_control`、实际使用的 TRL 接口清单和训练指标；`sft_manifest.json` 记录样本构成、
学习率、轮数和参数变化验证。这两份文件是判断"是否真的训练过"的依据。

注意：`GRPO_OUTPUT_ROOT` 只在**设置它的那条命令**里生效。评估时同样要带上，否则指标会写回项目
目录而不是持久化目录。

评估时若 `MODEL_PATH` 指向含 `adapter_config.json` 的目录，加载器会自动按 LoRA adapter
处理，并从 `base_model_name_or_path` 取回基座模型，因此第 3 步的 adapter 目录可以直接用于评估。

轨迹文件每行一条任务，字段为 `task_id`、`prompt`、`turns`、`final_text`、`termination`、
`verification`、`rewards`；模型解析失败时对应 turn 会带 `parse_error`。

## 上传百度 AI Studio

推荐按以下顺序操作：

1. 上传代码，不上传模型权重、checkpoint、日志或访问凭据。
2. 将模型单独上传或挂载到持久化目录。
3. 进入项目根目录，检查 Python、PyTorch、CUDA 和 GPU。
4. 安装缺失依赖，避免无必要地覆盖平台 PyTorch。
5. 设置 `MODEL_PATH` 和 `GRPO_OUTPUT_ROOT`。
6. 依次运行单元测试、Oracle smoke、模型加载测试和 `--dry-run`。
7. 只在以上步骤通过后执行少量真实训练步。
8. 检查 `run_manifest.json`、global step、参数变化和 checkpoint 可重新加载性。
9. 再扩大数据量和训练步数。

代码没有硬编码 Windows 盘符，核心入口使用 `pathlib`，同一份代码可在 Windows 和 Linux 下运行。

## 常见问题

- **缺少可选依赖**：在隔离环境安装训练依赖。
- **CUDA OOM**：减少候选数、batch 和生成长度，增加梯度累积，并启用 LoRA/gradient checkpointing。
- **离线模型加载失败**：检查 `MODEL_PATH` 是否包含完整 Transformers 模型文件。
- **所有组奖励相同**：该组没有 GRPO 相对优势信号。先确认模型是否能输出合法协议（跑一次
  `--policy model` 评估并查看 `logs/traces`，输出被 markdown 围栏包裹即为此类问题），再依次
  启用冷启动 SFT 和 `reward.format` 部分信用。
- **`Trainer ran but no trainable parameter value changed`**：训练跑完但梯度为零，通常是上一条
  导致的零方差，也可能是学习率被 scheduler 归零，先检查日志里的 `reward_std` 与 `learning_rate`。
- **解析失败**：检查 JSONL 轨迹，确认模型每轮只输出一个合法 JSON 对象。
- **TRL 参数不存在**：目标版本与配置不兼容；记录实际版本，不要静默忽略配置项。

## 验收状态术语

- `PASS`：已实际运行并通过。
- `PARTIAL`：部分实现或仅通过部分测试。
- `NOT RUN`：代码存在但尚未执行。
- `BLOCKED`：受硬件、网络、模型或依赖限制无法验证。

必须分别报告：环境测试通过、模型推理通过、工具调用通过、Trainer 启动成功、真实参数更新完成、训练后评估完成。前一个状态不能代替后一个状态。
