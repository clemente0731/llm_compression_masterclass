# LLM Compression Masterclass: Tutorial Lectures

本文档包含完整的讲义内容，从数学原理到工程实现，逐步深入。

---

# Lecture 1: The "Why" and "How" of Compression (为什么需要压缩)

## 1.1 The Memory Wall Problem (内存墙问题)

### 1.1.1 现代 GPU 的困境

让我们从一个简单的计算开始：

**问题**: 为什么 Llama-3 70B 在单张 A100 80GB 上无法运行？

```
计算:
- 70B 参数
- FP16 精度: 每个参数 2 bytes
- 总大小: 70B × 2 = 140 GB
- A100 显存: 80 GB

结论: 140 GB > 80 GB, 无法放入显存
```

但这只是存储问题。更深层的问题是 **带宽瓶颈**。

### 1.1.2 推理是 Memory Bound 的

**关键公式: Arithmetic Intensity**

```
Arithmetic Intensity (AI) = FLOPs / Bytes

对于矩阵乘法 Y = X @ W:
- FLOPs: 2 × batch × seq_len × hidden × hidden
- Bytes: batch × seq_len × hidden × 2 (input) + hidden × hidden × 2 (weight)

当 batch=1, seq_len=1 (自回归生成):
- FLOPs ≈ 2 × hidden²
- Bytes ≈ 2 × hidden² (weight dominates)
- AI ≈ 1 FLOP/byte

这意味着: 每加载 1 byte 数据，只做 1 次浮点运算
GPU 的算力被严重浪费！
```

**Roofline Model 分析**:

```
                    Compute Bound
                    ↓
Throughput         ___________
(GFLOPS)          /
                 /   ← Memory Bound
                /
               /
              /|
             / |
            /  |
           /   |
          /    |
         /     |
        /      |
    ───┴───────────────────→
       ^                    Arithmetic Intensity (FLOP/Byte)
       |
       LLM 推理在这里！
       (AI ≈ 1)
```

### 1.1.3 压缩的本质

**如果我们把模型从 FP16 压缩到 INT4**:

```
原始:
- Weight size: 140 GB (FP16)
- Load time: 140 GB / 2 TB/s = 70 ms

压缩后:
- Weight size: 35 GB (INT4)
- Load time: 35 GB / 2 TB/s = 17.5 ms

加速比: 70 / 17.5 = 4x
```

**关键洞见**: 压缩不仅节省显存，还能直接加速推理！

## 1.2 Quantization Fundamentals (量化基础)

### 1.2.1 什么是量化？

量化是将连续的浮点数映射到离散的整数的过程。

```
原始 (FP16):
    x = [0.123, -0.456, 0.789, ...]  (每个数 16 bits)

量化后 (INT8):
    q = [12, -46, 79, ...]           (每个数 8 bits)
    scale = 0.01
    zero_point = 0

反量化:
    x_hat = scale * (q - zero_point)
          = 0.01 * [12, -46, 79, ...]
          = [0.12, -0.46, 0.79, ...]  (有误差！)
```

### 1.2.2 量化误差分析

**量化误差的来源**:

```
误差 = |x - x_hat|

对于 round-to-nearest:
    最大误差 = 0.5 × scale

scale 越大 → 量化范围越大 → 误差越大
scale 越小 → 量化范围越小 → 可能 overflow
```

**如何选择 scale？**

```python
# MinMax 方法
def calculate_scale(x, num_bits, symmetric=True):
    if symmetric:
        max_val = max(abs(x.min()), abs(x.max()))
        q_max = 2 ** (num_bits - 1) - 1
        scale = max_val / q_max
    else:
        q_max = 2 ** num_bits - 1
        scale = (x.max() - x.min()) / q_max
    return scale
```

### 1.2.3 量化粒度详解

**Per-Tensor Quantization**:
```
整个权重矩阵共享一个 scale

优点: 存储开销最小 (只需 1 个 scale)
缺点: 精度最低 (无法适应局部分布)

适用: 激活值量化 (Per-Tensor Static)
```

**Per-Channel Quantization**:
```
每个输出通道一个 scale

W 形状: [out_features, in_features]
scale 形状: [out_features, 1]

for each output channel i:
    scale[i] = calculate_scale(W[i, :])

优点: 更好地适应每个通道的分布
缺点: 需要存储 out_features 个 scale

适用: 权重量化 (INT8)
```

**Per-Group Quantization**:
```
每 group_size 个元素一个 scale

W 形状: [out_features, in_features]
假设 group_size = 128
scale 形状: [out_features, in_features // 128]

for each row i:
    for each group j:
        start = j * 128
        end = start + 128
        scale[i, j] = calculate_scale(W[i, start:end])

优点: 最高精度 (适应局部分布)
缺点: 存储开销较大

适用: 权重量化 (INT4), group_size=128 是常用值
```

### 1.2.4 Symmetric vs. Asymmetric

**Symmetric Quantization**:
```
假设数据以 0 为中心

公式:
    q = round(x / scale)
    x_hat = scale * q

特点:
    - zero_point = 0
    - 量化范围: [-scale * q_max, scale * q_max]
    - 适合: 权重 (通常以 0 为中心)
```

**Asymmetric Quantization**:
```
可以处理任意分布

公式:
    q = round(x / scale) + zero_point
    x_hat = scale * (q - zero_point)

特点:
    - zero_point != 0
    - 量化范围: [scale * (q_min - zero_point), scale * (q_max - zero_point)]
    - 适合: 激活值 (可能有偏移，如 ReLU 后)
```

**示例**:
```python
# Symmetric (权重)
weights = [-0.5, -0.2, 0.1, 0.3, 0.8]
# 范围: [-0.5, 0.8], 以 0 为中心取 max(0.5, 0.8) = 0.8
scale = 0.8 / 127  # INT8, q_max = 127
q = round(weights / scale)  # [-79, -32, 16, 47, 127]

# Asymmetric (激活值, 假设 ReLU 后)
activations = [0.0, 0.2, 0.5, 0.8, 1.0]
# 范围: [0.0, 1.0]
scale = 1.0 / 255  # UINT8, q_max = 255
zero_point = round(-0.0 / scale) = 0
q = round(activations / scale) + zero_point  # [0, 51, 127, 204, 255]
```

## 1.3 Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)

### 1.3.1 PTQ 工作流程

```
PTQ Pipeline:

1. 加载预训练模型 (FP16)
   ↓
2. 准备校准数据 (100-1000 samples)
   ↓
3. Forward Pass 收集统计信息
   - 权重: 直接计算 min/max
   - 激活: 运行校准数据，观测分布
   ↓
4. 计算量化参数 (scale, zero_point)
   ↓
5. 量化权重
   ↓
6. 保存量化模型
```

### 1.3.2 QAT 工作流程

```
QAT Pipeline:

1. 加载预训练模型 (FP16)
   ↓
2. 插入 Fake Quantization 节点
   - Forward: 模拟量化 (round + clip)
   - Backward: Straight-Through Estimator
   ↓
3. 使用完整数据集训练
   ↓
4. 模型"学习"适应量化误差
   ↓
5. 导出量化模型
```

### 1.3.3 对比

| 维度 | PTQ | QAT |
|------|-----|-----|
| 数据需求 | 少量校准数据 (100-1000 samples) | 完整训练数据集 |
| 时间成本 | 分钟~小时 | 天~周 |
| 计算成本 | 低 | 高 (需要 GPU 训练) |
| 精度恢复 | INT8 几乎无损, INT4 有挑战 | 最佳 |
| 适用场景 | 生产部署, 快速迭代 | 追求极致精度 |

**为什么 `llm-compressor` 选择 PTQ？**
1. LLM 训练成本极高，重新训练不现实
2. 对于大多数场景，PTQ + 高级算法 (GPTQ, AWQ) 已足够
3. 部署速度要求快

---

# Lecture 2: The `llm-compressor` Architecture (架构设计)

## 2.1 Design Philosophy (设计哲学)

### 2.1.1 声明式配置 (Declarative Configuration)

`llm-compressor` 采用 **Recipe Pattern**：

```python
# 命令式 (Imperative) - 不推荐
model = load_model()
for layer in model.layers:
    if is_linear(layer):
        layer.weight = quantize_gptq(layer.weight, calibration_data)
model.save()

# 声明式 (Declarative) - llm-compressor 的方式
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear"),
]
oneshot(model=model, recipe=recipe, dataset=calibration_data)
```

**优势**:
1. **可读性**: 一眼就能看出做了什么
2. **可复现性**: Recipe 可以保存、分享
3. **可组合性**: 多个 Modifier 可以链式组合
4. **解耦性**: 算法实现与用户配置分离

### 2.1.2 生命周期管理 (Lifecycle Management)

每个 Modifier 都遵循统一的生命周期：

```
┌─────────────────────────────────────────────────┐
│                 Modifier Lifecycle              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────┐     ┌──────────┐     ┌─────────┐ │
│  │initialize│ ──► │on_event()│ ──► │finalize │ │
│  └──────────┘     └──────────┘     └─────────┘ │
│       │                │                │      │
│       ▼                ▼                ▼      │
│  Setup Config    Handle Events    Cleanup      │
│  Register Hooks  Collect Stats    Save State   │
│                  Apply Changes                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 2.2 Core Components (核心组件)

### 2.2.1 `oneshot()` 函数

`oneshot()` 是用户的主要入口点：

```python
def oneshot(
    # 模型参数
    model: str | PreTrainedModel,  # 模型路径或实例
    tokenizer: str | PreTrainedTokenizerBase | None = None,
    
    # Recipe 参数
    recipe: str | list[Modifier] | None = None,  # 压缩配方
    
    # 数据集参数
    dataset: str | Dataset | None = None,  # 校准数据集
    num_calibration_samples: int = 512,
    max_seq_length: int = 384,
    
    # 输出参数
    output_dir: str | None = None,
    
    **kwargs,
) -> PreTrainedModel:
    """
    执行 one-shot 校准压缩。
    返回压缩后的模型。
    """
```

**内部流程**:

```python
# Simplified oneshot implementation
def oneshot(**kwargs):
    # 1. 解析参数
    model_args, dataset_args, recipe_args = parse_args(**kwargs)
    
    # 2. 预处理
    model, processor = pre_process(model_args, dataset_args)
    
    # 3. 准备校准数据
    dataloader = get_calibration_dataloader(dataset_args, processor)
    
    # 4. 初始化 Session
    session = active_session()
    session.initialize(
        model=model,
        recipe=recipe_args.recipe,
        calib_data=dataloader,
    )
    
    # 5. 选择并运行 Pipeline
    pipeline = CalibrationPipeline.from_modifiers(session.lifecycle.recipe.modifiers)
    pipeline(model, dataloader, dataset_args)
    
    # 6. 完成
    session.finalize()
    
    # 7. 保存
    post_process(model_args, recipe_args, output_dir)
    
    return model
```

### 2.2.2 `CompressionSession` 类

Session 管理全局压缩状态：

```python
class CompressionSession:
    """
    全局压缩会话管理器。
    
    职责:
    1. 管理 CompressionLifecycle
    2. 提供全局访问点 (active_session())
    3. 处理并发和资源管理
    """
    
    def initialize(self, model, recipe, **kwargs):
        """初始化会话和所有 Modifiers"""
        self.lifecycle.initialize(recipe=recipe, model=model, **kwargs)
    
    def finalize(self):
        """完成会话，清理资源"""
        self.lifecycle.finalize()
    
    def reset(self):
        """重置会话，用于多次压缩"""
        self.lifecycle.reset()
```

### 2.2.3 `CompressionLifecycle` 类

Lifecycle 负责调度 Modifier 的生命周期事件：

```python
@dataclass
class CompressionLifecycle:
    state: State
    recipe: Recipe
    
    def initialize(self, recipe, **kwargs):
        """
        初始化流程:
        1. 解析 Recipe，创建 Modifier 实例
        2. 调用每个 Modifier 的 on_initialize()
        """
        self.recipe = Recipe.create_instance(recipe)
        for mod in self.recipe.modifiers:
            mod.initialize(state=self.state, **kwargs)
    
    def event(self, event_type: EventType, **kwargs):
        """
        事件分发:
        将事件传递给所有 Modifiers
        """
        event = Event(type_=event_type)
        for mod in self.recipe.modifiers:
            mod.update_event(state=self.state, event=event, **kwargs)
    
    def finalize(self):
        """
        完成流程:
        调用每个 Modifier 的 on_finalize()
        """
        for mod in self.recipe.modifiers:
            mod.finalize(state=self.state)
```

### 2.2.4 Event System (事件系统)

```python
class EventType(Enum):
    """
    事件类型枚举。
    
    训练生命周期 (用于 QAT):
        INITIALIZE, FINALIZE
        BATCH_START, LOSS_CALCULATED, BATCH_END
        OPTIM_PRE_STEP, OPTIM_POST_STEP
    
    校准生命周期 (用于 PTQ):
        CALIBRATION_EPOCH_START  - 校准开始
        SEQUENTIAL_EPOCH_END     - 一层校准完成 (Sequential Pipeline)
        CALIBRATION_EPOCH_END    - 校准结束
    """
    
    # PTQ 关键事件
    CALIBRATION_EPOCH_START = "calibration_epoch_start"
    SEQUENTIAL_EPOCH_END = "sequential_epoch_end"
    CALIBRATION_EPOCH_END = "calibration_epoch_end"
```

**事件流示例 (Sequential Pipeline)**:

```
时间 ─────────────────────────────────────────────────────────►

CALIBRATION_EPOCH_START
│
├── Layer 0-3 校准完成 ─► SEQUENTIAL_EPOCH_END ─► GPTQ 压缩
│
├── Layer 4-7 校准完成 ─► SEQUENTIAL_EPOCH_END ─► GPTQ 压缩
│
├── Layer 8-11 校准完成 ─► SEQUENTIAL_EPOCH_END ─► GPTQ 压缩
│
├── ...
│
└── 所有层完成 ─► CALIBRATION_EPOCH_END ─► 清理 Hooks
```

## 2.3 Modifier System (Modifier 系统)

### 2.3.1 Modifier 基类

```python
class Modifier(ModifierInterface, HooksMixin):
    """
    所有 Modifier 的基类。
    
    关键属性:
        initialized_: 是否已初始化
        finalized_: 是否已完成
        started_: 是否已开始
        ended_: 是否已结束
    
    生命周期方法:
        on_initialize(): 初始化时调用
        on_start(): 开始时调用
        on_event(): 处理事件
        on_end(): 结束时调用
        on_finalize(): 完成时调用
    """
    
    @abstractmethod
    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        初始化 Modifier。
        
        典型实现:
        1. 解析配置
        2. 准备数据结构
        3. 可选: 注册初始 Hooks
        
        返回: True 如果成功
        """
        raise NotImplementedError()
    
    def on_start(self, state: State, event: Event, **kwargs):
        """
        Modifier 开始时调用。
        
        典型实现:
        1. 注册校准 Hooks
        2. 初始化统计收集器
        """
        pass
    
    def on_event(self, state: State, event: Event, **kwargs):
        """
        处理事件。
        
        典型实现:
        根据 event.type_ 执行不同操作
        """
        pass
    
    def on_end(self, state: State, event: Event, **kwargs):
        """
        Modifier 结束时调用。
        
        典型实现:
        1. 移除 Hooks
        2. 冻结量化参数
        """
        pass
    
    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        清理资源。
        
        典型实现:
        1. 清除临时数据
        2. 验证状态
        
        返回: True 如果成功
        """
        return True
```

### 2.3.2 HooksMixin

Hooks 是 PyTorch 的机制，用于在 forward/backward pass 中注入自定义逻辑：

```python
class HooksMixin:
    """
    提供 Hook 管理功能的 Mixin。
    
    方法:
        register_hook(): 注册 hook
        remove_hooks(): 移除所有 hooks
        disable_hooks(): 临时禁用 hooks (context manager)
    """
    
    _hooks: list[RemovableHandle] = []
    
    def register_hook(
        self,
        module: torch.nn.Module,
        hook_fn: Callable,
        hook_type: str = "forward",  # "forward", "forward_pre", "backward"
        with_kwargs: bool = False,
    ):
        """
        注册 hook。
        
        Args:
            module: 目标模块
            hook_fn: hook 函数
            hook_type: hook 类型
            with_kwargs: 是否包含 kwargs
        """
        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "forward_pre":
            handle = module.register_forward_pre_hook(
                hook_fn, with_kwargs=with_kwargs
            )
        elif hook_type == "backward":
            handle = module.register_full_backward_hook(hook_fn)
        
        self._hooks.append(handle)
    
    def remove_hooks(self):
        """移除所有已注册的 hooks"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
    
    @classmethod
    @contextmanager
    def disable_hooks(cls):
        """临时禁用所有 hooks"""
        # ... implementation
```

### 2.3.3 Modifier 继承层次

```
Modifier (base)
├── QuantizationModifier      # 简单 PTQ
│   └── QuantizationMixin     # 量化配置管理
│
├── GPTQModifier              # GPTQ 算法
│   └── QuantizationMixin
│
├── AWQModifier               # AWQ 算法
│   └── QuantizationMixin
│
├── SmoothQuantModifier       # SmoothQuant 算法
│
├── SparseGPTModifier         # 稀疏化
│
├── WandaModifier             # Wanda 剪枝
│
├── QuIPModifier              # Hadamard 变换
│
└── SpinQuantModifier         # SpinQuant 变换
```

## 2.4 Pipeline System (Pipeline 系统)

### 2.4.1 Pipeline 选择逻辑

```python
class CalibrationPipeline(ABC, RegistryMixin):
    @classmethod
    def from_modifiers(cls, modifiers, user=None):
        """
        根据 Modifiers 自动选择合适的 Pipeline。
        
        逻辑:
        1. 如果只有 QuantizationModifier 且不需要校准数据 → DataFree
        2. 否则 → Sequential (默认)
        3. 用户可以通过 user 参数覆盖
        """
        inferred = cls._infer_pipeline(modifiers)
        pipeline = user or inferred
        return cls.load_from_registry(pipeline)
    
    @staticmethod
    def _infer_pipeline(modifiers):
        # 只有一个 QuantizationModifier 且不需要校准数据
        if len(modifiers) == 1 and isinstance(modifiers[0], QuantizationModifier):
            config = modifiers[0].resolve_quantization_config()
            if not config.requires_calibration_data():
                return "datafree"
        return "sequential"
```

### 2.4.2 Sequential Pipeline 详解

```python
@CalibrationPipeline.register("sequential")
class SequentialPipeline(CalibrationPipeline):
    """
    逐层处理的 Pipeline，适合大模型。
    
    优点:
    1. 内存高效: 只需保持一层的激活缓存
    2. 支持大模型: 可以处理无法完整放入显存的模型
    3. 准确: 使用量化后的输出作为下一层的输入
    
    缺点:
    1. 需要模型可追踪 (torch.fx)
    2. 某些动态操作可能不支持
    """
    
    @staticmethod
    def __call__(model, dataloader, dataset_args):
        """
        执行逐层校准。
        
        步骤:
        1. 追踪模型，划分 subgraph
        2. 对每个 subgraph:
           a. Pass 1: 触发 hooks，收集统计
           b. 触发 SEQUENTIAL_EPOCH_END，应用压缩
           c. Pass 2: 获取压缩后的输出
           d. 缓存输出，作为下一个 subgraph 的输入
        3. 触发 CALIBRATION_EPOCH_END
        """
        
        # Step 1: Trace and partition
        subgraphs = trace_subgraphs(model, sample_input, sequential_targets)
        
        # Step 2: Start calibration
        LifecycleCallbacks.calibration_epoch_start()
        
        # Step 3: Prepare activation cache
        activations = IntermediatesCache.from_dataloader(dataloader)
        
        # Step 4: Process each subgraph
        for subgraph in subgraphs:
            
            # Pass 1: Calibration (triggers hooks)
            for batch_idx in range(len(dataloader)):
                inputs = activations.fetch(batch_idx, subgraph.input_names)
                subgraph.forward(model, **inputs)
            
            # Apply compression for this subgraph
            LifecycleCallbacks.sequential_epoch_end(subgraph)
            
            # Pass 2: Propagation (no hooks, just get outputs)
            with HooksMixin.disable_hooks():
                for batch_idx in range(len(dataloader)):
                    inputs = activations.fetch(batch_idx, subgraph.input_names)
                    output = subgraph.forward(model, **inputs)
                    
                    # Cache outputs for next subgraph
                    activations.update(batch_idx, output)
                    activations.delete(batch_idx, subgraph.consumed_names)
        
        # Step 5: End calibration
        LifecycleCallbacks.calibration_epoch_end()
```

---

# Lecture 3: Deep Dive into GPTQ (GPTQ 算法详解)

## 3.1 Mathematical Foundation (数学基础)

### 3.1.1 问题定义

**目标**: 找到量化后的权重矩阵 \( Q \)，使得输出误差最小化。

```
原始计算: Y = X @ W
量化计算: Y_hat = X @ Q

目标: min_Q || Y - Y_hat ||^2
     = min_Q || X @ W - X @ Q ||^2
     = min_Q || X @ (W - Q) ||^2
```

### 3.1.2 Layer-wise 分解

由于直接优化整个模型的 \( Q \) 不可行，GPTQ 采用 **layer-wise** 策略：

```
对每一层独立优化:
    min_Q_l || X_l @ W_l - X_l @ Q_l ||^2

其中 X_l 是校准数据经过前 l-1 层后的激活值
```

### 3.1.3 Hessian Matrix

定义损失函数:
```
L(Q) = || X @ (W - Q) ||^2
     = (W - Q)^T @ X^T @ X @ (W - Q)
     = (W - Q)^T @ H @ (W - Q)

其中 H = X^T @ X 是 Hessian 矩阵 (实际上是 Fisher Information 的近似)
```

**Hessian 的意义**:
- \( H_{ii} \) 大 → 第 i 个权重对输出影响大 → 需要保护
- \( H_{ij} \) 大 → 第 i 和第 j 个权重相关性强 → 可以相互补偿

### 3.1.4 Optimal Brain Surgeon (OBS) 理论

GPTQ 基于 OBS 的核心公式：

```
当我们量化权重 w_i 到 q_i 时，引入误差 δ_i = w_i - q_i

为了最小化整体误差，其他权重应该调整:
    Δw = -δ_i * H^{-1}[:, i] / H^{-1}[i, i]

这就是 "误差补偿" 的数学基础
```

## 3.2 Algorithm Walkthrough (算法流程)

### 3.2.1 GPTQ 核心算法

```
Algorithm: GPTQ (one layer)
Input: Weight W, Hessian H, quantization config
Output: Quantized weight Q

1. Compute inverse Hessian: H_inv = inverse(H)
2. For each column i = 1 to n:
   a. Quantize: q_i = quantize(w_i)
   b. Compute error: δ_i = w_i - q_i
   c. Update remaining weights:
      W[:, i+1:] -= δ_i * H_inv[i, i+1:] / H_inv[i, i]
3. Return Q
```

### 3.2.2 源码分析

**Step 1: Hessian 累积**

```python
# src/llmcompressor/modifiers/quantization/gptq/gptq_quantize.py

def accumulate_hessian(inp, module, H, num_samples):
    """
    累积 Hessian 矩阵。
    
    Args:
        inp: 输入激活值 [batch, seq_len, hidden]
        module: Linear 层
        H: 当前 Hessian 矩阵
        num_samples: 已处理的样本数
    
    Returns:
        更新后的 (H, num_samples)
    """
    # Reshape input for Linear layer
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    
    if isinstance(module, torch.nn.Linear):
        # [batch * seq_len, hidden] -> [hidden, batch * seq_len]
        inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
    
    # Update count
    num_added = inp.shape[1]
    
    # Moving average update
    H *= num_samples / (num_samples + num_added)
    num_samples += num_added
    
    # Accumulate: H += inp @ inp.T (normalized)
    inp = inp.to(dtype=torch.float32)
    inp = math.sqrt(2 / num_samples) * inp
    H += inp.matmul(inp.t())
    
    return H, num_samples
```

**Step 2: Cholesky 分解**

```python
def quantize_weight(module, quant_args, hessians_dict, blocksize=128, percdamp=0.01):
    """
    使用 GPTQ 量化权重。
    """
    W = module.weight.clone()
    H = hessians_dict[module]
    
    # =====================================
    # Step 2a: Add damping for stability
    # =====================================
    # Damping prevents numerical instability when H is near-singular
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    
    # =====================================
    # Step 2b: Cholesky decomposition
    # =====================================
    # H = L @ L^T
    # H^{-1} = (L^{-1})^T @ L^{-1}
    # Cholesky(H^{-1}) = upper triangular
    try:
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        # Fallback: use identity (no error propagation)
        logger.warning("Hessian inversion failed, falling back to RTN")
        Hinv = torch.eye(num_columns, dtype=H.dtype, device=H.device)
    
    # ... continue with quantization
```

**为什么用 Cholesky？**

```
直接计算 H^{-1} 的问题:
1. 数值不稳定
2. 计算复杂度 O(n^3)

Cholesky 的优势:
1. 数值稳定 (利用正定性)
2. 计算高效
3. 可以逐列访问 (适合 block update)

关键性质:
- 如果 H = L @ L^T (Cholesky 分解)
- 则 H^{-1} = (L^{-1})^T @ L^{-1}
- upper(cholesky(H^{-1})) 给出我们需要的结构
```

**Step 3: 逐列量化 + 误差补偿**

```python
    # =====================================
    # Step 3: Block-wise quantization
    # =====================================
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        count = i2 - i1
        
        W1 = W[:, i1:i2].clone()  # Current block
        Q1 = torch.zeros_like(W1)  # Quantized block
        Err1 = torch.zeros_like(W1)  # Error block
        Hinv1 = Hinv[i1:i2, i1:i2]  # Corresponding Hessian inverse
        
        # Process each column in the block
        for i in range(count):
            w = W1[:, i]  # Current column
            d = Hinv1[i, i]  # Diagonal element
            q = w.clone()
            
            # =====================================
            # Step 3a: Quantize current column
            # =====================================
            if strategy == QuantizationStrategy.TENSOR:
                q = fake_quantize(q, scale, zero_point, quant_args)
            elif strategy == QuantizationStrategy.CHANNEL:
                q = fake_quantize(q, scale[:, 0], zero_point[:, 0], quant_args)
            elif strategy == QuantizationStrategy.GROUP:
                group_index = (i1 + i) // group_size
                q = fake_quantize(q, scale[:, group_index], 
                                  zero_point[:, group_index], quant_args)
            
            Q1[:, i] = q
            
            # =====================================
            # Step 3b: Compute and propagate error
            # =====================================
            # error = (original - quantized) / diagonal
            err = (w - q) / d
            
            # Update remaining columns in block
            # W1[:, i+1:] -= err @ Hinv1[i, i+1:]
            w1_err = err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            W1[:, i:] -= w1_err
            
            Err1[:, i] = err
        
        # Apply block update
        W[:, i1:i2] = Q1
        
        # Propagate error to remaining columns outside block
        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        W[:, i2:] -= w_err
    
    return loss, W, scale, zero_point, g_idx
```

### 3.2.3 Activation Ordering

GPTQ 支持多种列排序策略：

```python
# Activation Ordering Options
class ActivationOrdering(Enum):
    STATIC = "static"   # 按 Hessian 对角线排序
    GROUP = "group"     # 先分组，组内排序
    WEIGHT = "weight"   # 先排序，再分组
    NONE = None         # 不排序

def _apply_activation_ordering(W, H):
    """
    按激活值大小排序权重列。
    
    思想: 
    - Hessian 对角线 H[i,i] 表示第 i 列的重要性
    - 优先量化不重要的列 (H[i,i] 小的)
    - 让重要的列有更多机会被误差补偿
    """
    # Sort by diagonal (descending)
    perm = torch.argsort(torch.diag(H), descending=True)
    
    # Permute weight and Hessian
    return W[:, perm], H[perm][:, perm], perm
```

**为什么要 Activation Ordering？**

```
示例:
假设 H = [[10, 0],
          [0, 1]]

表示:
- 第 0 列很重要 (H[0,0] = 10)
- 第 1 列不重要 (H[1,1] = 1)

如果按顺序量化:
1. 量化列 0 → 误差只能补偿到列 1
2. 量化列 1 → 没有后续列可以补偿

如果按 H 排序 (先量化不重要的):
1. 量化列 1 → 列 0 可以补偿
2. 量化列 0 → 误差较小 (已经被补偿)

结果: 排序后整体误差更小!
```

## 3.3 GPTQModifier 实现

```python
class GPTQModifier(Modifier, QuantizationMixin):
    """
    GPTQ 量化 Modifier。
    
    生命周期:
    1. on_initialize: 初始化量化配置，准备模块映射
    2. on_start: 注册 Hessian 累积 hooks
    3. on_event(SEQUENTIAL_EPOCH_END): 执行 compress_modules
    4. on_end: 移除 hooks
    5. on_finalize: 清理 Hessian 数据
    """
    
    # GPTQ 参数
    block_size: int = 128           # 每次处理的列数
    dampening_frac: float = 0.01    # Hessian 阻尼系数
    actorder: str = "static"        # 激活排序策略
    offload_hessians: bool = False  # 是否卸载 Hessian 到 CPU
    
    # 内部状态
    _hessians: Dict[Module, Tensor]  # 模块 -> Hessian 矩阵
    _num_samples: Dict[Module, int]  # 模块 -> 样本计数
    
    def calibrate_module(self, module, args, _output):
        """
        Hessian 累积 hook。
        
        在每次 forward pass 时被调用，累积该模块的 Hessian。
        """
        inp = args[0]
        
        # Initialize Hessian if not present
        if module not in self._num_samples:
            self._hessians[module] = make_empty_hessian(module)
            self._num_samples[module] = 0
        
        # Accumulate
        self._hessians[module], self._num_samples[module] = accumulate_hessian(
            inp, module, self._hessians[module], self._num_samples[module]
        )
    
    def compress_modules(self):
        """
        对所有已校准的模块应用 GPTQ。
        
        在 SEQUENTIAL_EPOCH_END 事件时被调用。
        """
        for module in list(self._num_samples.keys()):
            quant_args = module.quantization_scheme.weights
            
            # Apply GPTQ
            loss, quantized_weight, scale, zero_point, g_idx = quantize_weight(
                module=module,
                quant_args=quant_args,
                hessians_dict=self._hessians,
                blocksize=self.block_size,
                percdamp=self.dampening_frac,
            )
            
            # Update module parameters
            update_offload_parameter(module, "weight", quantized_weight)
            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)
            if g_idx is not None:
                update_offload_parameter(module, "weight_g_idx", g_idx)
            
            # Cleanup
            del self._num_samples[module]
```

---

# Lecture 4: Exercises & Self-Assessment (练习与自测)

## 4.1 Exercise 1: Basic FP8 Quantization

**目标**: 理解最简单的量化流程

**代码**: 见 `exercise_1_hello_world.py`

**要点检查**:
- [ ] 理解 `QuantizationModifier` 的作用
- [ ] 理解 `scheme="FP8_DYNAMIC"` 的含义
- [ ] 理解为什么 `ignore=["lm_head"]` 是最佳实践

## 4.2 Exercise 2: GPTQ with Custom Dataset

**目标**: 使用自定义校准数据集进行 GPTQ 量化

**代码**: 见 `exercise_2_mixed_precision.py`

**要点检查**:
- [ ] 理解 `targets` 参数如何控制量化范围
- [ ] 理解 `GPTQModifier` 和 `QuantizationModifier` 的区别
- [ ] 理解 `group_size` 对精度和大小的影响

## 4.3 Exercise 3: Model Inspection

**目标**: 验证量化是否真的生效

**代码**: 见 `exercise_3_inspection.py`

**要点检查**:
- [ ] 理解如何检查模块的 dtype
- [ ] 理解量化后模型的结构变化
- [ ] 理解 `weight_scale` 和 `weight_zero_point` 属性

---

## Self-Assessment Quiz (自测题)

### Question 1: Fundamentals

**Q1**: 为什么 LLM 推理是 Memory Bound 而不是 Compute Bound？

<details>
<summary>Answer</summary>

因为 LLM 推理的 Arithmetic Intensity (AI) 很低：

```
AI = FLOPs / Bytes ≈ 1

对于 batch_size=1 的自回归生成:
- 每加载 2 bytes (FP16 weight)
- 只做 2 FLOPs (乘加)
- AI = 2 / 2 = 1

GPU 的算力/带宽比通常 > 100
所以瓶颈是带宽，不是算力
```
</details>

### Question 2: Quantization Theory

**Q2**: Per-Group Quantization (group_size=128) 相比 Per-Channel 有什么优劣？

<details>
<summary>Answer</summary>

**优势**:
- 更高精度: 每 128 个元素一个 scale，更好适应局部分布
- 对 INT4 特别重要: 因为量化范围太小，需要更细粒度

**劣势**:
- 存储开销: 需要 `num_weights / 128` 个 scale，而不是 `out_features` 个
- 计算开销: 反量化时需要更多操作

**经验法则**:
- INT8: Per-Channel 通常足够
- INT4: Per-Group (128) 是标配
</details>

### Question 3: GPTQ Algorithm

**Q3**: GPTQ 中的 Cholesky 分解为什么重要？不用会怎样？

<details>
<summary>Answer</summary>

**Cholesky 分解的作用**:
1. **数值稳定性**: 直接求 H^{-1} 可能因为 H 病态而不稳定
2. **计算效率**: Cholesky 是 O(n³) 但常数小，且可以逐列访问
3. **结构利用**: 上三角结构让 block update 更高效

**不用 Cholesky 的后果**:
```python
# llm-compressor 的 fallback 策略
except torch._C._LinAlgError:
    logger.warning("Hessian inversion failed, falling back to RTN")
    Hinv = torch.eye(num_columns)  # 恒等矩阵
```

当 fallback 到恒等矩阵时:
- 误差补偿失效 (Hinv[i, j] = 0 for i != j)
- 退化为 Round-to-Nearest (RTN)
- 精度显著下降
</details>

### Question 4: Architecture

**Q4**: `llm-compressor` 中 Sequential Pipeline 相比 Basic Pipeline 的优势是什么？

<details>
<summary>Answer</summary>

**Sequential Pipeline 优势**:

1. **内存效率**:
   ```
   Basic: 需要完整的激活缓存 (所有层)
   Sequential: 只需当前层和下一层的激活缓存
   ```

2. **大模型支持**:
   ```
   Basic: 模型必须完全放入 GPU
   Sequential: 可以配合 offloading 处理超大模型
   ```

3. **量化准确性**:
   ```
   Basic: 使用原始激活值校准后续层
   Sequential: 使用量化后的激活值校准后续层 (更真实)
   ```

**Sequential Pipeline 劣势**:
1. 需要模型可追踪 (torch.fx)
2. 某些动态操作可能不支持
3. 实现更复杂
</details>

### Question 5: Practical

**Q5**: 如果量化后模型输出 NaN，你会如何排查和解决？

<details>
<summary>Answer</summary>

**排查步骤**:

1. **检查校准数据**:
   ```python
   for batch in dataloader:
       assert not torch.isnan(batch).any()
       assert not torch.isinf(batch).any()
   ```

2. **检查原始模型**:
   ```python
   with torch.no_grad():
       output = model(sample_input)
       assert not torch.isnan(output).any()
   ```

3. **增加 Hessian 阻尼**:
   ```python
   recipe = GPTQModifier(dampening_frac=0.1)  # 从 0.01 增加到 0.1
   ```

4. **增加校准样本**:
   ```python
   oneshot(..., num_calibration_samples=1024)  # 从 512 增加
   ```

5. **检查特定层**:
   ```python
   # 逐层检查是哪一层出问题
   for name, module in model.named_modules():
       if hasattr(module, 'weight_scale'):
           if torch.isnan(module.weight_scale).any():
               print(f"NaN scale in {name}")
   ```

6. **排除敏感层**:
   ```python
   recipe = GPTQModifier(
       ignore=["lm_head", "model.layers.0", "model.layers.31"]
   )
   ```
</details>

---

## Further Reading (延伸阅读)

### Papers
1. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
2. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
3. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
4. [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)
5. [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304)

### Documentation
- [llm-compressor Official Docs](https://docs.vllm.ai/projects/llm-compressor/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [compressed-tensors Library](https://github.com/neuralmagic/compressed-tensors)

### Source Code
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
