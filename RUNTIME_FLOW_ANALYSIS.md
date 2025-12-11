# LLM-Compressor 运行时调度流程分析

> 本文档深入分析 llm-compressor 项目在运行时的执行流程、事件调度机制、Hook 系统和内存管理策略。

---

## 目录

1. [整体运行时架构](#1-整体运行时架构)
2. [Session 管理与线程安全](#2-session-管理与线程安全)
3. [生命周期事件流](#3-生命周期事件流)
4. [Modifier 状态机](#4-modifier-状态机)
5. [Hook 系统详解](#5-hook-系统详解)
6. [Pipeline 执行流程](#6-pipeline-执行流程)
7. [内存管理与 Offloading](#7-内存管理与-offloading)
8. [数据流跟踪](#8-数据流跟踪)
9. [运行时序列图](#9-运行时序列图)
10. [调试与监控](#10-调试与监控)

---

## 1. 整体运行时架构

### 1.1 核心组件关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Code: oneshot()                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Oneshot Orchestrator                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ pre_process  │→ │  initialize  │→ │   pipeline   │→ │ post_process │    │
│  │ (model load) │  │  (session)   │  │  (calibrate) │  │ (save model) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │ CompressionSess│  │ CalibrationPipe│  │ LifecycleCallba│
        │     ion        │  │     line       │  │     cks        │
        └────────────────┘  └────────────────┘  └────────────────┘
                │                   │                   │
                ▼                   ▼                   ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │ CompressionLife│  │ IntermediatesC │  │    Event       │
        │     cycle      │  │     ache       │  │   Dispatch     │
        └────────────────┘  └────────────────┘  └────────────────┘
                │                   
                ▼                   
        ┌────────────────────────────────────────┐
        │              Modifiers                 │
        │  ┌──────────┐ ┌──────────┐ ┌────────┐ │
        │  │ Quantize │ │   GPTQ   │ │  AWQ   │ │
        │  │ Modifier │ │ Modifier │ │Modifier│ │
        │  └──────────┘ └──────────┘ └────────┘ │
        └────────────────────────────────────────┘
```

### 1.2 调用入口追踪

当用户调用 `oneshot()` 函数时，执行流程如下：

```python
# file: src/llmcompressor/entrypoints/oneshot.py

def oneshot(...) -> PreTrainedModel:
    """
    entry point for one-shot compression
    """
    local_args = {k: v for k, v in locals().items() if k not in ("local_args", "kwargs")}
    one_shot = Oneshot(**local_args, **kwargs)  # step 1: instantiate
    one_shot()                                    # step 2: execute
    return one_shot.model
```

---

## 2. Session 管理与线程安全

### 2.1 全局 Session 与线程本地存储

```python
# file: src/llmcompressor/core/session_functions.py

_global_session = CompressionSession()      # global fallback
_local_storage = threading.local()          # thread-local storage
_local_storage.session = _global_session    # initialize

def active_session() -> CompressionSession:
    """
    returns the active session for current thread
    
    thread safety: each thread can have its own session via _local_storage
    """
    global _local_storage
    return getattr(_local_storage, "session", _global_session)
```

### 2.2 Session 创建上下文管理器

```python
@contextmanager
def create_session() -> Generator[CompressionSession, None, None]:
    """
    context manager for creating a new isolated session
    
    usage:
        with create_session() as session:
            # session is thread-local, isolated from global session
            session.initialize(...)
            # ...
        # original session restored after exiting context
    """
    global _local_storage
    orig_session = getattr(_local_storage, "session", None)
    new_session = CompressionSession()
    _local_storage.session = new_session
    try:
        yield new_session
    finally:
        _local_storage.session = orig_session  # restore original
```

### 2.3 Session 状态图

```
                      ┌──────────────────┐
                      │     CREATED      │
                      │  (initial state) │
                      └────────┬─────────┘
                               │ reset() / __init__()
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                         UNINITIALIZED                            │
│                    initialized_ = False                          │
│                    finalized = False                             │
└──────────────────────────────┬───────────────────────────────────┘
                               │ initialize()
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                          INITIALIZED                             │
│                    initialized_ = True                           │
│                    finalized = False                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    EVENT LOOP                               ││
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  ││
│  │  │ BATCH   │ →  │  LOSS   │ →  │ OPTIM   │ →  │ BATCH   │  ││
│  │  │ START   │    │ CALCUL  │    │ STEP    │    │  END    │  ││
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────┬───────────────────────────────────┘
                               │ finalize()
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                          FINALIZED                               │
│                    initialized_ = True                           │
│                    finalized = True                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 生命周期事件流

### 3.1 EventType 枚举

```python
# file: src/llmcompressor/core/events/event.py

class EventType(Enum):
    # lifecycle events (cannot be invoked via event())
    INITIALIZE = "initialize"
    FINALIZE = "finalize"
    
    # training loop events (standard order)
    BATCH_START = "batch_start"
    LOSS_CALCULATED = "loss_calculated"
    OPTIM_PRE_STEP = "optim_pre_step"
    OPTIM_POST_STEP = "optim_post_step"
    BATCH_END = "batch_end"
    
    # calibration-specific events
    CALIBRATION_EPOCH_START = "calibration_epoch_start"
    CALIBRATION_EPOCH_END = "calibration_epoch_end"
    SEQUENTIAL_EPOCH_END = "sequential_epoch_end"
```

### 3.2 事件顺序验证

```python
# file: src/llmcompressor/core/lifecycle.py

class CompressionLifecycle:
    # valid event order for training loop
    _event_order: list[EventType] = field(
        default_factory=lambda: [
            EventType.BATCH_START,
            EventType.LOSS_CALCULATED,
            EventType.OPTIM_PRE_STEP,
            EventType.OPTIM_POST_STEP,
            EventType.BATCH_END,
        ]
    )
    
    def _validate_event_order(self, event_type: EventType) -> bool:
        """
        ensures events are called in the correct order
        
        prevents:
            - calling BATCH_START twice without BATCH_END
            - calling LOSS_CALCULATED before BATCH_START
            - etc.
        """
        if event_type not in self._event_order:
            return True  # calibration events bypass validation
        
        if event_type == EventType.BATCH_START:
            valid = self._last_event_type != EventType.BATCH_START
        else:
            last_event_index = self._event_order.index(self._last_event_type)
            curr_event_index = self._event_order.index(event_type)
            valid = last_event_index <= curr_event_index
        
        if valid:
            self._last_event_type = event_type
        return valid
```

### 3.3 事件分发流程

```
┌────────────────────────────────────────────────────────────────────┐
│                    LifecycleCallbacks.event()                      │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                    active_session().event()                        │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                    lifecycle.event(event_type)                     │
│                                                                    │
│   1. validate: initialized_ == True, finalized == False            │
│   2. validate: event_type not in [INITIALIZE, FINALIZE]            │
│   3. validate: _validate_event_order(event_type)                   │
│   4. update global_step                                            │
│   5. create Event object                                           │
│   6. dispatch to all modifiers                                     │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│     for mod in self.recipe.modifiers:                              │
│         data = mod.update_event(state=self.state, event=event)     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Modifier 状态机

### 4.1 Modifier 生命周期状态

```python
# file: src/llmcompressor/modifiers/modifier.py

class Modifier:
    # state flags
    initialized_: bool = False   # set by on_initialize
    finalized_: bool = False     # set by on_finalize
    started_: bool = False       # set by on_start
    ended_: bool = False         # set by on_end
```

### 4.2 状态转换图

```
                    ┌───────────────────────────────────────┐
                    │              CREATED                  │
                    │  initialized_=False, started_=False   │
                    │  finalized_=False, ended_=False       │
                    └───────────────────┬───────────────────┘
                                        │ initialize()
                                        │ → on_initialize()
                                        ▼
              ┌─────────────────────────────────────────────────────────┐
              │                    INITIALIZED                          │
              │              initialized_=True                          │
              │                                                         │
              │   if self.start is set and start <= event.current_index │
              │   ───────────────────────────────────────────────────── │
              │                        │ on_start()                     │
              │                        ▼                                │
              │   ┌───────────────────────────────────────┐             │
              │   │              STARTED                  │             │
              │   │          started_=True                │             │
              │   │                                       │             │
              │   │   EVENT LOOP: on_event(), on_update() │             │
              │   │                                       │             │
              │   │   if self.end is set and end <= curr  │             │
              │   │   ─────────────────────────────────── │             │
              │   │                │ on_end()             │             │
              │   │                ▼                      │             │
              │   │   ┌────────────────────────┐          │             │
              │   │   │       ENDED            │          │             │
              │   │   │    ended_=True         │          │             │
              │   │   └────────────────────────┘          │             │
              │   └───────────────────────────────────────┘             │
              └─────────────────────────────────────────────────────────┘
                                        │ finalize()
                                        │ → on_finalize()
                                        ▼
                    ┌───────────────────────────────────────┐
                    │              FINALIZED                │
                    │           finalized_=True             │
                    └───────────────────────────────────────┘
```

### 4.3 update_event 方法详解

```python
def update_event(self, state: State, event: Event, **kwargs):
    """
    core event handler - dispatches to appropriate lifecycle methods
    
    call order:
    1. on_event (always called for all events)
    2. on_start (if BATCH_START and should_start)
    3. on_update (if started and not ended)
    4. on_end (if BATCH_END and should_end)
    """
    if not self.initialized_:
        raise RuntimeError("Cannot update an uninitialized modifier")
    
    if self.finalized_:
        raise RuntimeError("Cannot update a finalized modifier")
    
    # always dispatch to on_event first
    self.on_event(state, event, **kwargs)
    
    # handle starting the modifier if needed
    if (
        event.type_ == EventType.BATCH_START
        and not self.started_
        and self.should_start(event)
    ):
        self.on_start(state, event, **kwargs)
        self.started_ = True
        self.on_update(state, event, **kwargs)
        return
    
    # handle ending the modifier if needed
    if (
        event.type_ == EventType.BATCH_END
        and not self.ended_
        and self.should_end(event)
    ):
        self.on_end(state, event, **kwargs)
        self.ended_ = True
        self.on_update(state, event, **kwargs)
        return
    
    # regular update if started but not ended
    if self.started_ and not self.ended_:
        self.on_update(state, event, **kwargs)
```

---

## 5. Hook 系统详解

### 5.1 HooksMixin 类

```python
# file: src/llmcompressor/modifiers/utils/hooks.py

class HooksMixin(BaseModel):
    """
    mixin providing hook management capabilities:
    - register hooks with automatic disable support
    - global disable/enable mechanism
    - per-modifier hook tracking
    """
    
    # class-level (global) state
    _HOOKS_DISABLED: ClassVar[bool] = False           # global disable flag
    _HOOKS_KEEP_ENABLED: ClassVar[Set[RemovableHandle]] = set()  # exceptions
    
    # instance-level state
    _hooks: Set[RemovableHandle] = set()  # hooks registered by this instance
```

### 5.2 Hook 注册流程

```python
def register_hook(
    self,
    target: Union[torch.nn.Module, torch.nn.Parameter],
    hook: Callable[[Any], Any],
    hook_type: str,
    **kwargs,
) -> RemovableHandle:
    """
    registers a hook with global disable support
    
    hook_type mappings:
    - "forward" -> module.register_forward_hook(hook)
    - "forward_pre" -> module.register_forward_pre_hook(hook)
    - "full_backward" -> module.register_full_backward_hook(hook)
    - "query"/"key"/"value" -> compressed_tensors special hooks
    """
    handle = None
    
    @wraps(hook)
    def wrapped_hook(*args, **kwargs):
        nonlocal handle
        
        # check global disable state
        if (
            HooksMixin._HOOKS_DISABLED
            and handle not in HooksMixin._HOOKS_KEEP_ENABLED
        ):
            return  # skip hook execution
        
        return hook(*args, **kwargs)
    
    # get appropriate register function
    register_function = self._get_register_function(target, hook_type)
    handle = register_function(wrapped_hook, **kwargs)
    self._hooks.add(handle)
    
    return handle
```

### 5.3 Hook 禁用上下文

```python
@classmethod
@contextlib.contextmanager
def disable_hooks(cls, keep: Set[RemovableHandle] = frozenset()):
    """
    context manager to temporarily disable all hooks
    
    usage:
        # disable all hooks
        with HooksMixin.disable_hooks():
            model.forward(...)  # no hooks fire
        
        # disable all hooks except specific ones
        with HooksMixin.disable_hooks(keep={handle1, handle2}):
            model.forward(...)  # only handle1, handle2 fire
    """
    try:
        cls._HOOKS_DISABLED = True
        cls._HOOKS_KEEP_ENABLED |= keep
        yield
    finally:
        cls._HOOKS_DISABLED = False
        cls._HOOKS_KEEP_ENABLED -= keep
```

### 5.4 GPTQ 的 Hook 使用示例

```python
# file: src/llmcompressor/modifiers/quantization/gptq/base.py

class GPTQModifier(Modifier, QuantizationMixin):
    def on_start(self, state: State, event: Event, **kwargs):
        # 1. register quantization calibration hooks
        QuantizationMixin.start_calibration(self, state.model)
        
        # 2. register GPTQ-specific hessian accumulation hooks
        for _, module in match_named_modules(state.model, self.resolved_targets, self.ignore):
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # register forward hook to accumulate hessian
                self.register_hook(module, self.calibrate_module, "forward")
    
    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        forward hook: accumulates hessian from input activations
        
        called during calibration forward pass
        """
        inp = args[0]  # assume first argument is the input
        
        # initialize hessian if not present
        if module not in self._num_samples:
            init_device = "cpu" if self.offload_hessians else get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0
        
        # accumulate hessian
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp, module, self._hessians[module], self._num_samples[module]
            )
```

### 5.5 Hook 类型汇总

| Hook Type | PyTorch Method | 触发时机 | 用途 |
|-----------|----------------|----------|------|
| `forward_pre` | `register_forward_pre_hook` | forward 前 | 输入激活校准 |
| `forward` | `register_forward_hook` | forward 后 | 输出激活校准、Hessian 累积 |
| `full_backward` | `register_full_backward_hook` | backward 后 | 梯度分析 |
| `query` | 自定义 | Attention Q 计算后 | KV-cache 量化 |
| `key` | 自定义 | Attention K 计算后 | KV-cache 量化 |
| `value` | 自定义 | Attention V 计算后 | KV-cache 量化 |

---

## 6. Pipeline 执行流程

### 6.1 Pipeline 选择逻辑

```python
# file: src/llmcompressor/pipelines/registry.py

class CalibrationPipeline:
    @classmethod
    def from_modifiers(cls, modifiers: List[Modifier], user: str | None = None):
        """
        selects appropriate pipeline based on modifiers and user preference
        
        decision logic:
        1. if user specifies pipeline, use it
        2. if any modifier requires sequential (GPTQ, AWQ, etc.), use sequential
        3. otherwise, use basic
        """
        if user is not None:
            return cls.load_from_registry(user)
        
        requires_sequential = any(
            getattr(mod, "sequential_targets", None) is not None
            for mod in modifiers
        )
        
        if requires_sequential:
            return cls.load_from_registry("sequential")
        else:
            return cls.load_from_registry("basic")
```

### 6.2 BasicPipeline 流程

```python
# file: src/llmcompressor/pipelines/basic/pipeline.py

@CalibrationPipeline.register("basic")
class BasicPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(model, dataloader, dataset_args):
        """
        simple calibration: forward pass through entire model
        
        flow:
        1. dispatch model for generation (move to GPU)
        2. send CALIBRATION_EPOCH_START event
        3. for each batch: forward pass (triggers hooks)
        4. send CALIBRATION_EPOCH_END event
        """
        dispatch_for_generation(model)
        model_device = get_execution_device(model)
        
        LifecycleCallbacks.calibration_epoch_start()
        
        with calibration_forward_context(model):
            for batch in tqdm.tqdm(dataloader, desc="Calibrating"):
                batch = apply_pad_mask_to_batch(batch)
                batch = tensors_to_device(batch, model_device)
                model(**batch)  # hooks fire here
        
        LifecycleCallbacks.calibration_epoch_end()
```

```
BasicPipeline 执行流程:

┌──────────────────────────────────────────────────────────────────┐
│                     calibration_epoch_start                      │
└───────────────────────────────┬──────────────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           ▼                           │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │                 Batch Loop                      │  │
    │  │                                                 │  │
    │  │   for batch in dataloader:                      │  │
    │  │       batch → GPU                               │  │
    │  │       model(**batch)                            │  │
    │  │           │                                     │  │
    │  │           ├─ forward_pre_hook (input calib)     │  │
    │  │           ├─ module.forward()                   │  │
    │  │           └─ forward_hook (output calib)        │  │
    │  │                                                 │  │
    │  └─────────────────────────────────────────────────┘  │
    │                                                       │
    └───────────────────────────┬───────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     calibration_epoch_end                        │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 SequentialPipeline 流程

```python
# file: src/llmcompressor/pipelines/sequential/pipeline.py

@CalibrationPipeline.register("sequential")
class SequentialPipeline(CalibrationPipeline):
    @staticmethod
    def __call__(model, dataloader, dataset_args):
        """
        sequential calibration: process model layer-by-layer
        
        flow:
        1. partition model into subgraphs by sequential_targets
        2. for each subgraph:
           a. calibration pass (hooks enabled)
           b. SEQUENTIAL_EPOCH_END event (triggers compression)
           c. propagation pass (hooks disabled, capture compressed outputs)
        """
        session = active_session()
        
        # prepare model
        dispatch_for_sequential(model)
        model_device = get_execution_device(model)
        
        # trace subgraphs
        modifiers = session.lifecycle.recipe.modifiers
        sequential_targets = get_sequential_targets(modifiers, model, dataset_args)
        sample_input = next(iter(dataloader))
        subgraphs = trace_subgraphs(model, sample_input, sequential_targets, ignore)
        
        LifecycleCallbacks.calibration_epoch_start()
        
        # prepare intermediates cache
        activations = IntermediatesCache.from_dataloader(dataloader, model_device)
        
        for subgraph_index, subgraph in enumerate(subgraphs):
            with disable_offloading():
                # PASS 1: calibration (hooks enabled)
                for batch_idx in tqdm(range(len(dataloader)), desc="Calibrating"):
                    inputs = activations.fetch(batch_idx, subgraph.input_names)
                    subgraph.forward(model, **inputs)  # hooks fire
                
                LifecycleCallbacks.sequential_epoch_end(subgraph)  # compress!
                
                # PASS 2: propagation (hooks disabled)
                with HooksMixin.disable_hooks():
                    for batch_idx in tqdm(range(len(dataloader)), desc="Propagating"):
                        inputs = activations.fetch(batch_idx, subgraph.input_names)
                        output = subgraph.forward(model, **inputs)
                        
                        if subgraph_index < num_subgraphs - 1:
                            activations.update(batch_idx, output)
                            activations.delete(batch_idx, subgraph.consumed_names)
        
        LifecycleCallbacks.calibration_epoch_end()
```

```
SequentialPipeline 执行流程:

┌────────────────────────────────────────────────────────────────────────┐
│                        Model Partitioning                              │
│                                                                        │
│   Model: [Embed] → [Layer0] → [Layer1] → ... → [LayerN] → [LMHead]    │
│                         ↓          ↓                ↓                  │
│   Subgraphs:      [Subgraph0] [Subgraph1] ...  [SubgraphN]            │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     calibration_epoch_start                            │
└────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               ▼                               │
    │   for each subgraph:                                          │
    │                                                               │
    │   ┌─────────────────────────────────────────────────────────┐ │
    │   │ PASS 1: Calibration (hooks ENABLED)                     │ │
    │   │                                                         │ │
    │   │   for batch_idx in batches:                             │ │
    │   │       inputs = cache.fetch(batch_idx)                   │ │
    │   │       subgraph.forward(model, **inputs)                 │ │
    │   │           │                                             │ │
    │   │           ├─ input_hook → accumulate statistics         │ │
    │   │           └─ forward_hook → accumulate Hessian          │ │
    │   │                                                         │ │
    │   └─────────────────────────────────────────────────────────┘ │
    │                               │                               │
    │                               ▼                               │
    │   ┌─────────────────────────────────────────────────────────┐ │
    │   │ sequential_epoch_end                                    │ │
    │   │                                                         │ │
    │   │   GPTQModifier.compress_modules()                       │ │
    │   │       → quantize_weight() for each module               │ │
    │   │       → update weight, scale, zero_point                │ │
    │   │                                                         │ │
    │   └─────────────────────────────────────────────────────────┘ │
    │                               │                               │
    │                               ▼                               │
    │   ┌─────────────────────────────────────────────────────────┐ │
    │   │ PASS 2: Propagation (hooks DISABLED)                    │ │
    │   │                                                         │ │
    │   │   with HooksMixin.disable_hooks():                      │ │
    │   │       for batch_idx in batches:                         │ │
    │   │           inputs = cache.fetch(batch_idx)               │ │
    │   │           outputs = subgraph.forward(model, **inputs)   │ │
    │   │           cache.update(batch_idx, outputs)              │ │
    │   │           cache.delete(batch_idx, consumed_names)       │ │
    │   │                                                         │ │
    │   │   NOTE: outputs now reflect compressed weights          │ │
    │   │                                                         │ │
    │   └─────────────────────────────────────────────────────────┘ │
    │                                                               │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      calibration_epoch_end                             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 内存管理与 Offloading

### 7.1 IntermediatesCache 设计

```python
# file: src/llmcompressor/pipelines/cache.py

class IntermediatesCache:
    """
    caches intermediate activations between subgraphs
    
    key features:
    - automatic offloading to CPU when stored
    - automatic onloading to GPU when fetched
    - supports nested dataclass offloading
    - memory tracking via size() method
    """
    
    batch_intermediates: List[IntermediateValues]  # per-batch storage
    offload_device: Optional[torch.device]         # where to offload (usually CPU)
```

### 7.2 Offload/Onload 流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STORE (offload)                             │
│                                                                     │
│   tensor on GPU ──→ IntermediateValue ──→ stored in cache (CPU)    │
│        │                   │                                        │
│        │                   │                                        │
│        └── device=cuda     ├── value = tensor.to(cpu)              │
│                            └── device = cuda (saved for restore)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         FETCH (onload)                              │
│                                                                     │
│   IntermediateValue ──→ tensor on GPU                              │
│        │                      │                                     │
│        ├── value (on CPU)     │                                     │
│        └── device = cuda ─────┴──→ return value.to(device=cuda)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 递归 Offloading 支持

```python
@classmethod
def _offload_value(cls, value, offload_device, onload_device=None):
    """
    recursively offloads nested structures
    
    supported types:
    - torch.Tensor → offload to CPU
    - list/tuple → recurse
    - dict → recurse
    - dataclass → recurse through fields
    - primitives → pass through
    """
    match value:
        case torch.Tensor():
            return IntermediateValue(
                value=value.to(device=offload_device),
                device=(onload_device if onload_device else value.device),
            )
        case list():
            return IntermediateValue(
                value=[cls._offload_value(v, ...) for v in value],
                device=None,
            )
        case tuple():
            return IntermediateValue(
                value=tuple(cls._offload_value(v, ...) for v in value),
                device=None,
            )
        case dict():
            return IntermediateValue(
                value={k: cls._offload_value(v, ...) for k, v in value.items()},
                device=None,
            )
        case _ if is_dataclass(value):
            for field in fields(value):
                v = getattr(value, field.name)
                setattr(value, field.name, cls._offload_value(v, ...))
            return IntermediateValue(value=value, device=None)
        case _:
            return IntermediateValue(value=value, device=None)
```

### 7.4 Hessian Offloading (GPTQ)

```python
# file: src/llmcompressor/modifiers/quantization/gptq/base.py

class GPTQModifier:
    offload_hessians: bool = False  # user-configurable
    
    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module):
        """
        temporarily onloads hessian to execution device if offloading is enabled
        
        usage:
            with self._maybe_onload_hessian(module):
                # hessian is on GPU
                accumulate_hessian(...)
            # hessian back on CPU
        """
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)
        
        yield
        
        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted
                self._hessians[module] = self._hessians[module].to(device="cpu")
```

### 7.5 内存使用估算

```python
def size(self) -> Dict[torch.device, int]:
    """
    returns memory usage by device in bytes
    
    example return:
    {
        torch.device('cpu'): 1073741824,   # 1 GB on CPU
        torch.device('cuda:0'): 0,         # 0 bytes on GPU (all offloaded)
    }
    """
    sizes = defaultdict(lambda: 0)
    
    def _size_helper(intermediate):
        value = intermediate.value
        match value:
            case torch.Tensor():
                sizes[value.device] += value.nbytes
            case list() | tuple():
                for v in value:
                    _size_helper(v)
            case dict():
                for v in value.values():
                    _size_helper(v)
            case _ if is_dataclass(value):
                for field in fields(value):
                    _size_helper(getattr(value, field.name))
            case _:
                sizes[torch.device("cpu")] += sys.getsizeof(value, 0)
    
    for intermediates in self.batch_intermediates:
        for value in intermediates.values():
            _size_helper(value)
    
    return dict(sizes)
```

---

## 8. 数据流跟踪

### 8.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

1. DataLoader → IntermediatesCache (initial data)
   ┌────────────────┐        ┌────────────────────────────────┐
   │   DataLoader   │ ────→  │     IntermediatesCache         │
   │  (HF Dataset)  │        │                                │
   └────────────────┘        │  batch_0: {input_ids, attn_mask│
                             │  batch_1: {...}                │
                             │  ...                           │
                             │  batch_N: {...}                │
                             └────────────────────────────────┘

2. Cache → Subgraph → Cache (per subgraph)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ Subgraph 0 (Embedding Layer)                                             │
   │                                                                          │
   │  Cache.fetch(batch_idx, ["input_ids", "attention_mask"])                │
   │         │                                                                │
   │         ▼                                                                │
   │  ┌─────────────────────────────────────────────────────────────────────┐│
   │  │  subgraph_0.forward(model, input_ids=..., attention_mask=...)       ││
   │  │         │                                                           ││
   │  │         ├─ model.embed_tokens(input_ids)                            ││
   │  │         │         │                                                 ││
   │  │         │         └─ output: hidden_states                          ││
   │  │         │                                                           ││
   │  │         └─ return {"hidden_states": hidden_states}                  ││
   │  └─────────────────────────────────────────────────────────────────────┘│
   │         │                                                                │
   │         ▼                                                                │
   │  Cache.update(batch_idx, {"hidden_states": ...})                        │
   │  Cache.delete(batch_idx, ["input_ids"])  # consumed, free memory        │
   └──────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────────┐
   │ Subgraph 1 (Layer 0)                                                     │
   │                                                                          │
   │  Cache.fetch(batch_idx, ["hidden_states", "attention_mask", ...])       │
   │         │                                                                │
   │         ▼                                                                │
   │  ┌─────────────────────────────────────────────────────────────────────┐│
   │  │  [CALIBRATION PASS - hooks enabled]                                 ││
   │  │  subgraph_1.forward(model, hidden_states=..., ...)                  ││
   │  │         │                                                           ││
   │  │         ├─ forward_pre_hook: calibrate input activations            ││
   │  │         ├─ model.layers[0](hidden_states)                           ││
   │  │         ├─ forward_hook: accumulate Hessian (GPTQ)                  ││
   │  │         └─ output: new_hidden_states                                ││
   │  └─────────────────────────────────────────────────────────────────────┘│
   │         │                                                                │
   │         ▼                                                                │
   │  ┌─────────────────────────────────────────────────────────────────────┐│
   │  │  sequential_epoch_end: compress weights                             ││
   │  │         │                                                           ││
   │  │         └─ GPTQModifier.compress_modules()                          ││
   │  │                 │                                                   ││
   │  │                 └─ for each linear in subgraph:                     ││
   │  │                         quantize_weight(linear, hessian, ...)       ││
   │  │                         update_offload_parameter(linear, "weight")  ││
   │  └─────────────────────────────────────────────────────────────────────┘│
   │         │                                                                │
   │         ▼                                                                │
   │  ┌─────────────────────────────────────────────────────────────────────┐│
   │  │  [PROPAGATION PASS - hooks disabled]                                ││
   │  │  subgraph_1.forward(model, hidden_states=..., ...)                  ││
   │  │         │                                                           ││
   │  │         └─ output: new_hidden_states (with compressed weights)      ││
   │  └─────────────────────────────────────────────────────────────────────┘│
   │         │                                                                │
   │         ▼                                                                │
   │  Cache.update(batch_idx, {"hidden_states": new_hidden_states})          │
   │  Cache.delete(batch_idx, consumed_names)                                │
   └──────────────────────────────────────────────────────────────────────────┘

   ... repeat for remaining subgraphs ...
```

### 8.2 Subgraph 追踪系统

```python
# file: src/llmcompressor/pipelines/sequential/helpers.py

@dataclass
class Subgraph:
    """
    executable subgraph of model graph
    
    attributes:
    - graph: torch.fx.Graph representing operations
    - input_names: names of inputs required from cache
    - consumed_names: names that can be deleted after this subgraph
    """
    graph: Graph
    input_names: Set[str]      # what we need
    consumed_names: Set[str]   # what we can delete after use
    _code: Optional[PythonCode] = None  # cached compiled code
    
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        execute subgraph operations
        
        lazy compiles graph to Python code on first call
        """
        if self._code is None:
            self._code = self.graph.python_code("self")
            exec(self._code.src, self._code.globals)
        
        forward_fn = self._code.globals.get("forward")
        return forward_fn(*args, **kwargs)
```

---

## 9. 运行时序列图

### 9.1 GPTQ 完整执行序列

```
┌────────┐ ┌──────────┐ ┌────────────┐ ┌───────────────┐ ┌────────────┐ ┌───────────┐
│ User   │ │ Oneshot  │ │  Session   │ │ GPTQModifier  │ │  Pipeline  │ │   Cache   │
└───┬────┘ └────┬─────┘ └─────┬──────┘ └──────┬────────┘ └─────┬──────┘ └─────┬─────┘
    │          │              │               │                │              │
    │ oneshot()│              │               │                │              │
    │─────────>│              │               │                │              │
    │          │              │               │                │              │
    │          │ initialize() │               │                │              │
    │          │─────────────>│               │                │              │
    │          │              │               │                │              │
    │          │              │ on_initialize │                │              │
    │          │              │──────────────>│                │              │
    │          │              │               │                │              │
    │          │              │               │ apply_quant_   │              │
    │          │              │               │ config()       │              │
    │          │              │               │                │              │
    │          │              │               │<───────────────│              │
    │          │              │               │                │              │
    │          │ pipeline()   │               │                │              │
    │          │─────────────────────────────────────────────>│              │
    │          │              │               │                │              │
    │          │              │               │                │ from_data    │
    │          │              │               │                │ loader()     │
    │          │              │               │                │─────────────>│
    │          │              │               │                │              │
    │          │              │ calib_epoch   │                │              │
    │          │              │ _start        │                │              │
    │          │              │<──────────────────────────────│              │
    │          │              │               │                │              │
    │          │              │ on_start()    │                │              │
    │          │              │──────────────>│                │              │
    │          │              │               │                │              │
    │          │              │               │ register       │              │
    │          │              │               │ hooks()        │              │
    │          │              │               │                │              │
    │          │              │               │                │              │
    │          │              │               │                │ SUBGRAPH     │
    │          │              │               │                │ LOOP         │
    │          │              │               │                │              │
    │  ┌───────│──────────────│───────────────│────────────────│──────────────│──────┐
    │  │       │              │               │                │ fetch()      │      │
    │  │       │              │               │                │<────────────>│      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │ calibrate      │              │      │
    │  │       │              │               │ _module()      │              │      │
    │  │       │              │               │<───────────────│              │      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │ accumulate     │              │      │
    │  │       │              │               │ _hessian()     │              │      │
    │  │       │              │               │                │              │      │
    │  │       │              │ seq_epoch_end │                │              │      │
    │  │       │              │<──────────────────────────────│              │      │
    │  │       │              │               │                │              │      │
    │  │       │              │ on_event      │                │              │      │
    │  │       │              │ (SEQ_END)     │                │              │      │
    │  │       │              │──────────────>│                │              │      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │ compress       │              │      │
    │  │       │              │               │ _modules()     │              │      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │ quantize       │              │      │
    │  │       │              │               │ _weight()      │              │      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │                │ [PROPAGATE]  │      │
    │  │       │              │               │                │ (hooks off)  │      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │                │ update()     │      │
    │  │       │              │               │                │<────────────>│      │
    │  │       │              │               │                │              │      │
    │  │       │              │               │                │ delete()     │      │
    │  │       │              │               │                │<────────────>│      │
    │  └───────│──────────────│───────────────│────────────────│──────────────│──────┘
    │          │              │               │                │              │
    │          │              │ calib_epoch   │                │              │
    │          │              │ _end          │                │              │
    │          │              │<──────────────────────────────│              │
    │          │              │               │                │              │
    │          │              │ on_end()      │                │              │
    │          │              │──────────────>│                │              │
    │          │              │               │                │              │
    │          │              │               │ remove_hooks() │              │
    │          │              │               │                │              │
    │          │ finalize()   │               │                │              │
    │          │─────────────>│               │                │              │
    │          │              │               │                │              │
    │          │              │ on_finalize() │                │              │
    │          │              │──────────────>│                │              │
    │          │              │               │                │              │
    │          │ save()       │               │                │              │
    │          │              │               │                │              │
    │<─────────│              │               │                │              │
    │  model   │              │               │                │              │
```

### 9.2 事件时序图

```
Time ─────────────────────────────────────────────────────────────────────────────────>

Session:
    │ initialize()                                              finalize()
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
Events:
    │         CALIB_EPOCH_START                    CALIB_EPOCH_END│
    │              │                                    │         │
    │              ├── [Subgraph 0] ──────────────┬────│         │
    │              │         │                    │    │         │
    │              │     SEQ_EPOCH_END ───────────┘    │         │
    │              │                                    │         │
    │              ├── [Subgraph 1] ──────────────┬────│         │
    │              │         │                    │    │         │
    │              │     SEQ_EPOCH_END ───────────┘    │         │
    │              │                                    │         │
    │              ├── [Subgraph N] ──────────────┬────│         │
    │              │         │                    │    │         │
    │              │     SEQ_EPOCH_END ───────────┘    │         │
    │              │                                    │         │
    │              └───────────────────────────────────┘         │
    │                                                             │

Modifier State:
    │                                                             │
    ├ initialized ─────────────────────────────────────── finalized
    │         │                                      │
    │         started ──────────────────────── ended │
    │                   │                   │        │
    │                   │   [event loop]    │        │
    │                   │   on_event()      │        │
    │                   └───────────────────┘        │
```

---

## 10. 调试与监控

### 10.1 启用详细日志

```python
# via environment variable
import os
os.environ["LLM_COMPRESSOR_LOG_FILE"] = "/path/to/debug.log"

# via oneshot parameter
oneshot(
    model=model,
    recipe=recipe,
    log_dir="/path/to/logs"  # creates timestamped log file
)
```

### 10.2 Loguru 日志级别

```python
from loguru import logger

# 关键日志点 (已内置)
# - lifecycle.py: "Initializing compression lifecycle"
# - lifecycle.py: "Finalized modifier: {mod}"
# - hooks.py: "{self} added {handle}"
# - gptq/base.py: "Quantizing {name} using {num_samples} samples"
```

### 10.3 监控 Hook 状态

```python
# 检查 hook 是否激活
from llmcompressor.modifiers.utils.hooks import HooksMixin

print(f"Hooks disabled: {HooksMixin._HOOKS_DISABLED}")
print(f"Keep enabled: {HooksMixin._HOOKS_KEEP_ENABLED}")

# 检查 modifier hooks
modifier = session.lifecycle.recipe.modifiers[0]
print(f"Registered hooks: {modifier._hooks}")
```

### 10.4 监控内存使用

```python
# 检查 cache 内存使用
cache_size = activations.size()
for device, bytes_used in cache_size.items():
    print(f"{device}: {bytes_used / 1e9:.2f} GB")

# 检查 Hessian 内存使用 (GPTQ)
for module, hessian in gptq_modifier._hessians.items():
    print(f"{module}: {hessian.numel() * hessian.element_size() / 1e6:.2f} MB")
```

### 10.5 添加自定义监控 Hook

```python
def monitor_hook(module, args, output):
    """
    custom hook for debugging module execution
    """
    input_tensor = args[0]
    print(f"Module: {module.__class__.__name__}")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Input dtype: {input_tensor.dtype}")
    print(f"  Input device: {input_tensor.device}")
    print(f"  Output shape: {output.shape}")
    return output

# 注册到特定模块
model.layers[0].self_attn.q_proj.register_forward_hook(monitor_hook)
```

### 10.6 性能分析

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    oneshot(model=model, recipe=recipe, dataset=dataset)

# 导出 Chrome trace
prof.export_chrome_trace("oneshot_trace.json")

# 打印统计
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

---

## 总结

llm-compressor 的运行时架构采用了以下关键设计模式：

1. **Session/Lifecycle 模式**: 通过 `CompressionSession` 和 `CompressionLifecycle` 管理全局状态，支持多线程隔离
2. **Event-Driven 架构**: 使用事件系统解耦 Pipeline 和 Modifier，支持灵活的扩展
3. **Hook 系统**: 基于 PyTorch Hook 机制，支持全局禁用和选择性激活
4. **Pipeline 抽象**: BasicPipeline 和 SequentialPipeline 提供不同的校准策略
5. **智能内存管理**: IntermediatesCache 自动处理 GPU/CPU 数据迁移，降低显存压力
6. **状态机 Modifier**: 每个 Modifier 有清晰的生命周期状态转换

理解这些运行时机制对于调试问题、扩展功能和优化性能都非常重要。

