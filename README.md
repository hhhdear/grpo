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
- Clip-Higher、token-level loss、零方差组过滤和 overlong shaping 的纯 PyTorch 数值参考。
- 独立的最小 GRPO 参考公式及反向传播测试。

尚未在当前机器完成的部分：

- 当前环境缺少 Transformers、TRL、PEFT、Datasets、Accelerate 和模型权重，因此尚未完成真实 Qwen 推理、GRPO 参数更新或训练后评估。
- 当前 `dynamic_sampling()` 是零方差组过滤原语，不是带重采样预算的完整 Dynamic Sampling 训练循环。
- 尚未完成六组正式消融、三随机种子实验和 DAPO 相对 Vanilla GRPO 的研究结论。

详细审计见 [实现审计](docs/implementation_audit.md)。任何未运行实验都不得填写虚构数字。

## 项目结构

```text
grpo_agent/             订单环境、Agent、奖励、DAPO 数值核心
training/               TRL GRPO 训练入口和扩展接口
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

## 输出文件

- `data/generated/*.jsonl`：固定合成数据集；
- `logs/traces/*.jsonl`：模型输出、工具调用、状态变化和奖励分项；
- `logs/evaluation/*.json`：聚合指标及逐任务记录；
- `checkpoints/<run>/`：模型/adapter、tokenizer、trainer state 和运行清单。

设置 `GRPO_OUTPUT_ROOT` 后，checkpoint、轨迹和评估结果会写入指定持久化目录；数据集路径仍由 YAML 控制。

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
- **所有组奖励相同**：该组没有 GRPO 相对优势信号，需要检查任务和生成多样性。
- **解析失败**：检查 JSONL 轨迹，确认模型每轮只输出一个合法 JSON 对象。
- **TRL 参数不存在**：目标版本与配置不兼容；记录实际版本，不要静默忽略配置项。

## 验收状态术语

- `PASS`：已实际运行并通过。
- `PARTIAL`：部分实现或仅通过部分测试。
- `NOT RUN`：代码存在但尚未执行。
- `BLOCKED`：受硬件、网络、模型或依赖限制无法验证。

必须分别报告：环境测试通过、模型推理通过、工具调用通过、Trainer 启动成功、真实参数更新完成、训练后评估完成。前一个状态不能代替后一个状态。
