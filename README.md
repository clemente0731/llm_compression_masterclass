# LLM Compression Masterclass

这个目录包含了《LLM Compression Masterclass》教程的完整学习资料，包括深度理论讲义、项目源码解析和实战练习。

---

## Project Metadata

| Item | Value |
|------|-------|
| **基于项目** | llm-compressor |
| **Latest Commit** | `6cf8d29ca0cc790f92a6e2c3794d42aab3c396ff` |
| **Latest Tag** | `0.8.1` |
| **Commit Date** | 2025-12-10 |
| **Repository** | https://github.com/vllm-project/llm-compressor |

---

## 目录结构

```
llm_compression_masterclass/
│
├── README.md                          # 本文件 (学习指南)
│
├── LLM_COMPRESSOR_PROJECT_GUIDE.md    # 项目深度解析 (~1200行)
├── TUTORIAL_LECTURES.md               # 完整讲义 (~1300行)
├── RUNTIME_FLOW_ANALYSIS.md           # 运行时调度流程分析 (~800行)
│
├── DIR_GUIDE_examples.md              # examples/ 目录详解
├── DIR_GUIDE_llmcompressor.md         # src/llmcompressor/ 源码详解
├── DIR_GUIDE_tests.md                 # tests/ 目录详解
├── DIR_GUIDE_tools.md                 # tools/ 目录详解
│
├── exercise_1_hello_world.py          # 练习 1: 基础 FP8 量化
├── exercise_2_mixed_precision.py      # 练习 2: 混合精度 GPTQ 量化
└── exercise_3_inspection.py           # 练习 3: 量化结果验证工具
```

### 文档内容概览

| 文件 | 内容 |
|------|------|
| `LLM_COMPRESSOR_PROJECT_GUIDE.md` | 项目背景、技术原理、架构设计、源码分析、实战示例 |
| `TUTORIAL_LECTURES.md` | 4节讲义 + 5道自测题，从理论到实践 |
| `RUNTIME_FLOW_ANALYSIS.md` | Session管理、事件调度、Hook系统、Pipeline执行、内存管理 |
| `DIR_GUIDE_examples.md` | 19个示例目录的详细说明，按场景/硬件选择指南 |
| `DIR_GUIDE_llmcompressor.md` | 14个核心模块的源码结构和调用关系 |
| `DIR_GUIDE_tests.md` | 测试框架、测试类型、如何运行和添加测试 |
| `DIR_GUIDE_tools.md` | 环境收集工具和开发建议 |

---

## 学习路径 (推荐顺序)

### Phase 1: 理论基础 (2-3小时)

#### Step 1.1: 阅读项目指南
**文件**: `LLM_COMPRESSOR_PROJECT_GUIDE.md`

重点关注：
- Part 1: 理解 Memory Wall 问题和压缩的必要性
- Part 2: 掌握量化的数学原理 (Scale, Zero-Point, Granularity)
- Part 2.3: 深入理解 GPTQ, AWQ, SmoothQuant 算法

#### Step 1.2: 阅读教程讲义
**文件**: `TUTORIAL_LECTURES.md`

重点关注：
- Lecture 1: Arithmetic Intensity 和 Roofline Model
- Lecture 2: Recipe Pattern 和 Modifier Lifecycle
- Lecture 3: GPTQ 的 Hessian 计算和误差补偿机制

### Phase 2: 架构理解 (1-2小时)

**文件**: `LLM_COMPRESSOR_PROJECT_GUIDE.md` Part 3

重点关注：
- 3.2: Core Components (Oneshot, Session, Lifecycle)
- 3.2.3: Event System 的事件流
- 3.2.5: Sequential Pipeline 的工作原理

### Phase 2.5: 运行时流程深入 (1小时)

**文件**: `RUNTIME_FLOW_ANALYSIS.md`

重点关注：
- Section 2: Session 管理与线程本地存储
- Section 4: Modifier 状态机 (started_, ended_, finalized_)
- Section 5: Hook 系统与全局禁用机制
- Section 6: BasicPipeline vs SequentialPipeline 执行差异
- Section 7: IntermediatesCache 的 offload/onload 策略
- Section 9: GPTQ 完整执行序列图

### Phase 3: 动手实践 (2-3小时)

#### Step 3.1: Exercise 1 - Hello World
```bash
python exercise_1_hello_world.py
```
学习目标：
- 理解 `oneshot()` API
- 理解 `QuantizationModifier` 配置
- 理解 FP8 Dynamic 方案 (无需校准数据)

#### Step 3.2: Exercise 2 - Mixed Precision
```bash
python exercise_2_mixed_precision.py
```
学习目标：
- 理解 `GPTQModifier` 和校准数据
- 掌握 `targets` 参数的正则表达式用法
- 理解 MLP vs Attention 层的量化策略

#### Step 3.3: Exercise 3 - Inspection
```bash
python exercise_3_inspection.py ./tinyllama-fp8-dynamic
# 或
python exercise_3_inspection.py ./tinyllama-w4a16-mlp-only
```
学习目标：
- 理解量化模型的结构变化
- 掌握 `weight_scale`, `weight_zero_point`, `weight_g_idx`
- 学会验证量化效果

### Phase 4: 自测与巩固 (30分钟)

完成 `TUTORIAL_LECTURES.md` 中的 Self-Assessment Quiz：
- Question 1: Memory Bound 原理
- Question 2: Per-Group vs Per-Channel
- Question 3: Cholesky 分解的作用
- Question 4: Sequential vs Basic Pipeline
- Question 5: NaN 问题排查

---

## 环境准备

### 基础依赖
```bash
pip install llmcompressor transformers torch datasets
```

### 可选依赖 (用于测试 vLLM 推理)
```bash
pip install vllm
```

### 验证安装
```python
import llmcompressor
print(llmcompressor.__version__)  # 应显示 0.8.x
```

---

## 练习详情

### Exercise 1: Hello World of Compression
**文件**: `exercise_1_hello_world.py`
**目标**: 将 TinyLlama 模型量化为 FP8 格式
**难度**: ⭐ 入门
**时间**: ~10分钟

关键代码：
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"]
)
oneshot(model=model, recipe=recipe)
```

### Exercise 2: Mixed Precision Quantization
**文件**: `exercise_2_mixed_precision.py`
**目标**: 使用 GPTQ 对 MLP 层进行 INT4 量化
**难度**: ⭐⭐ 进阶
**时间**: ~20分钟

关键代码：
```python
from llmcompressor.modifiers.quantization import GPTQModifier

mlp_targets = [
    "model.layers.\\d+.mlp.gate_proj",
    "model.layers.\\d+.mlp.up_proj",
    "model.layers.\\d+.mlp.down_proj",
]
recipe = GPTQModifier(
    targets=mlp_targets,
    scheme="W4A16",
    ignore=["lm_head"],
)
oneshot(model=model, recipe=recipe, dataset=calibration_data)
```

### Exercise 3: Model Inspection
**文件**: `exercise_3_inspection.py`
**目标**: 检查量化后模型的权重结构
**难度**: ⭐⭐ 进阶
**时间**: ~5分钟

运行方式：
```bash
# 检查 Exercise 1 的结果
python exercise_3_inspection.py ./tinyllama-fp8-dynamic

# 检查 Exercise 2 的结果
python exercise_3_inspection.py ./tinyllama-w4a16-mlp-only
```

---

## 常见问题

### Q1: CUDA Out of Memory
```python
# 方案 1: 减少校准样本
oneshot(..., num_calibration_samples=128)

# 方案 2: 使用 CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto"
)

# 方案 3: 对于 GPTQ，offload Hessian
recipe = GPTQModifier(..., offload_hessians=True)
```

### Q2: 量化后输出 NaN
```python
# 方案 1: 增加阻尼系数
recipe = GPTQModifier(..., dampening_frac=0.1)

# 方案 2: 增加校准样本
oneshot(..., num_calibration_samples=1024)

# 方案 3: 排除更多敏感层
recipe = GPTQModifier(..., ignore=["lm_head", "model.layers.0"])
```

### Q3: 精度下降严重
```python
# 方案 1: 使用 SmoothQuant
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(...),
]

# 方案 2: 使用更大的 group_size
# (在 scheme 或 config_groups 中配置)

# 方案 3: 保持敏感层在更高精度
# (混合精度策略)
```

---

## 参考资源

### 论文
- [GPTQ](https://arxiv.org/abs/2210.17323): GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- [AWQ](https://arxiv.org/abs/2306.00978): AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- [SmoothQuant](https://arxiv.org/abs/2211.10438): SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

### 文档
- [llm-compressor 官方文档](https://docs.vllm.ai/projects/llm-compressor/)
- [vLLM 文档](https://docs.vllm.ai/)

### 代码
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)

---

## 反馈与问题

如果你在学习过程中遇到问题：
1. 检查 `LLM_COMPRESSOR_PROJECT_GUIDE.md` Part 7 的故障排除指南
2. 查看 llm-compressor 的 [GitHub Issues](https://github.com/vllm-project/llm-compressor/issues)
3. 加入 [vLLM Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) 的 `#llm-compressor` 频道

祝学习愉快！🚀
