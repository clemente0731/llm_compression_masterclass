# Directory Guide: tests/

这个文档详细解释 `llm-compressor/tests/` 目录的测试结构。

---

## 目录概览

```
tests/
├── __init__.py
├── testing_utils.py        # 测试工具函数
│
├── unit/                   # 单元测试
├── llmcompressor/          # 集成测试
├── e2e/                    # 端到端测试
├── examples/               # 示例测试
├── lmeval/                 # LM 评估测试
└── test_timer/             # 性能计时测试
```

---

## 详细说明

### 1. testing_utils.py - 测试工具

**用途**: 提供测试中常用的工具函数和 fixtures

**关键函数**:

```python
# 模型加载工具
def requires_gpu():
    """跳过没有 GPU 的测试"""

def requires_torch_version(min_version):
    """检查 PyTorch 版本"""

# 临时文件管理
@contextmanager
def temp_directory():
    """创建临时目录用于保存模型"""

# 模型比较
def compare_models(model1, model2, rtol=1e-3, atol=1e-5):
    """比较两个模型的权重"""

# 校准数据
def get_test_calibration_data(tokenizer, num_samples=8):
    """生成小规模测试用校准数据"""
```

---

### 2. unit/ - 单元测试

**用途**: 测试最小单元的功能

```
unit/
├── __init__.py
├── test_logger.py          # 日志功能测试
└── core/
    ├── __init__.py
    ├── test_state.py       # State 类测试
    └── events/
        ├── __init__.py
        └── test_event.py   # Event 类测试
```

**测试内容**:

| 文件 | 测试内容 |
|------|----------|
| `test_logger.py` | Loguru 日志配置 |
| `test_state.py` | State 数据类的序列化和操作 |
| `test_event.py` | Event 创建、属性计算、事件流 |

**示例测试**:

```python
# test_event.py
class TestEvent:
    def test_event_creation(self):
        event = Event(type_=EventType.CALIBRATION_EPOCH_START)
        assert event.type_ == EventType.CALIBRATION_EPOCH_START
    
    def test_epoch_calculation(self):
        event = Event(global_step=100, steps_per_epoch=10)
        assert event.epoch == 10
        assert event.epoch_step == 0
```

---

### 3. llmcompressor/ - 集成测试

**用途**: 测试各模块的集成功能

```
llmcompressor/
├── __init__.py
├── conftest.py             # pytest fixtures
├── helpers.py              # 测试辅助函数
│
├── metrics/                # 指标测试
│   ├── test_logger.py
│   └── utils/
│       └── test_frequency_manager.py
│
├── modeling/               # 模型特定测试
│   ├── test_calib_deepseek_v3.py
│   ├── test_calib_llama4.py
│   ├── test_calib_qwen3_next.py
│   ├── test_calib_qwen3_vl_moe.py
│   ├── test_calib_qwen3.py
│   └── test_fuse.py        # 权重融合测试
│
├── modifiers/              # Modifier 测试
│   ├── awq/
│   │   └── test_base.py    # AWQ 测试
│   ├── calibration/
│   │   ├── test_frozen.py  # 冻结状态测试
│   │   ├── test_lifecycle.py  # 生命周期测试
│   │   └── test_observers.py  # Observer 测试
│   ├── logarithmic_equalization/
│   │   └── test_base.py
│   ├── pruning/
│   │   ├── sparsegpt/
│   │   └── wanda/
│   ├── quantization/
│   │   ├── test_base.py
│   │   └── test_handling_shared_embeddings.py
│   ├── smoothquant/
│   │   ├── test_base.py
│   │   └── test_utils.py
│   ├── transform/
│   │   ├── test_correctness.py
│   │   └── test_serialization.py
│   └── utils/
│       └── test_hooks.py   # HooksMixin 测试
│
├── observers/              # Observer 测试
│   ├── test_helpers.py
│   ├── test_min_max.py
│   └── test_mse.py
│
├── pipelines/              # Pipeline 测试
│   ├── sequential/
│   ├── test_cache.py       # IntermediatesCache 测试
│   └── test_model_free_ptq.py
│
├── pytorch/                # PyTorch 工具测试
│   ├── helpers.py
│   ├── modifiers/          # (12 个测试文件)
│   └── utils/              # (3 个测试文件)
│
├── recipe/                 # Recipe 测试
│   └── test_recipe.py
│
├── test_sentinel.py        # Sentinel 测试
│
├── transformers/           # Transformers 集成测试
│   ├── autoround/
│   ├── compression/        # 压缩保存测试 (含 YAML 配置)
│   ├── data/               # 数据集测试
│   ├── gptq/
│   ├── kv_cache/
│   ├── oneshot/            # oneshot 测试
│   ├── sparsegpt/          # SparseGPT 测试
│   └── tracing/            # 追踪测试
│
└── utils/                  # 工具测试
    ├── pytorch/
    ├── test_helpers.py
    └── test_transformers.py
```

#### Modifier 测试重点

**AWQ 测试** (`modifiers/awq/test_base.py`):
```python
class TestAWQModifier:
    def test_awq_initialization(self):
        """测试 AWQ Modifier 初始化"""
    
    def test_awq_smoothing(self):
        """测试 AWQ 的 smoothing 计算"""
    
    def test_awq_grid_search(self):
        """测试最优 scale 的 grid search"""
```

**GPTQ 测试** (`modifiers/quantization/test_base.py`):
```python
class TestGPTQModifier:
    def test_hessian_accumulation(self):
        """测试 Hessian 矩阵累积"""
    
    def test_cholesky_decomposition(self):
        """测试 Cholesky 分解"""
    
    def test_error_propagation(self):
        """测试误差补偿"""
```

**Observer 测试** (`observers/test_min_max.py`):
```python
class TestMinMaxObserver:
    def test_memoryless_observer(self):
        """每次只看当前 batch"""
        observer = MemorylessMinMaxObserver(...)
        min1, max1 = observer.get_min_max(batch1)
        min2, max2 = observer.get_min_max(batch2)
        # min2, max2 只反映 batch2
    
    def test_static_observer(self):
        """记住全局 min/max"""
        observer = StaticMinMaxObserver(...)
        observer.get_min_max(batch1)
        min2, max2 = observer.get_min_max(batch2)
        # min2, max2 反映 batch1 和 batch2 的全局
```

---

### 4. e2e/ - 端到端测试

**用途**: 完整流程测试，从模型加载到推理验证

```
e2e/
├── __init__.py
├── e2e_utils.py            # E2E 测试工具
└── vLLM/                   # vLLM 集成测试
    ├── __init__.py
    ├── test_vllm.py        # 主测试文件
    ├── run_vllm.py         # vLLM 运行器
    │
    ├── configs/            # 量化配置
    │   ├── fp4_nvfp4.yaml
    │   ├── fp8_dynamic_per_token.yaml
    │   ├── fp8_block.yaml
    │   ├── int8_channel_weight_static_per_tensor_act.yaml
    │   ├── w4a16_grouped_quant.yaml
    │   ├── sparse_24.yaml
    │   └── ... (更多配置)
    │
    ├── recipes/            # 测试 Recipes
    │   ├── actorder/       # 激活排序测试
    │   ├── FP8/            # FP8 测试
    │   ├── INT8/           # INT8 测试
    │   ├── kv_cache/       # KV Cache 测试
    │   ├── Sparse_2of4/    # 稀疏测试
    │   ├── WNA16/          # W4A16/W8A16 测试
    │   └── WNA16_2of4/     # 稀疏+量化组合
    │
    ├── run_tests_in_python.sh   # Python 测试脚本
    └── run_tests_in_rhaiis.sh   # RHAIIS 测试脚本
```

#### E2E 测试流程

```python
# e2e/vLLM/test_vllm.py
class TestVLLMIntegration:
    @pytest.mark.parametrize("config", CONFIGS)
    def test_quantization_and_inference(self, config):
        """
        完整测试流程:
        1. 加载原始模型
        2. 应用量化配置
        3. 保存量化模型
        4. 用 vLLM 加载
        5. 验证推理输出
        """
        # Step 1: Load and quantize
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        oneshot(model=model, recipe=config["recipe"])
        model.save_pretrained(output_dir)
        
        # Step 2: Load with vLLM
        llm = LLM(model=output_dir)
        
        # Step 3: Verify inference
        output = llm.generate(prompts)
        assert output is not None
        assert len(output) > 0
```

#### 测试配置示例

**FP8 Dynamic** (`configs/fp8_dynamic_per_token.yaml`):
```yaml
recipe:
  - QuantizationModifier:
      targets: ["Linear"]
      scheme: "FP8_DYNAMIC"
      ignore: ["lm_head"]

test:
  model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  prompts: ["Hello, how are you?"]
  expected_output_length: 50
```

**W4A16 GPTQ** (`configs/w4a16_grouped_quant.yaml`):
```yaml
recipe:
  - GPTQModifier:
      targets: ["Linear"]
      scheme: "W4A16"
      ignore: ["lm_head"]
      block_size: 128

test:
  model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  calibration_samples: 128
```

---

### 5. examples/ - 示例测试

**用途**: 验证示例代码可以正确运行

```
examples/
├── __init__.py
├── test_example_scripts.py  # 测试示例脚本
└── test_readmes.py          # 测试 README 中的代码
```

**测试内容**:

| 测试 | 说明 |
|------|------|
| `test_example_scripts.py` | 运行 examples/ 目录中的脚本 |
| `test_readmes.py` | 提取并运行 README 中的代码块 |

**示例测试**:
```python
# test_example_scripts.py
class TestExampleScripts:
    @pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
    def test_example_runs_without_error(self, script):
        """验证示例脚本可以无错误运行"""
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            timeout=300,
        )
        assert result.returncode == 0
```

---

### 6. lmeval/ - LM 评估测试

**用途**: 使用 lm-evaluation-harness 评估量化模型

```
lmeval/
├── __init__.py
├── test_lmeval.py          # LM Eval 测试
└── configs/                # 评估配置
    ├── fp8_dynamic_per_token.yaml
    ├── fp8_static_per_tensor.yaml
    ├── int8_w8a8_dynamic_per_token.yaml
    ├── w4a16_actorder_group.yaml
    ├── w4a16_actorder_weight.yaml
    ├── w4a16_awq_sym.yaml
    ├── w4a4_nvfp4.yaml
    ├── vl_fp8_dynamic_per_token.yaml   # 视觉模型
    └── vl_int8_w8a8_dynamic_per_token.yaml
```

**评估流程**:
```python
# test_lmeval.py
def test_perplexity(config):
    """
    评估量化后模型的 perplexity:
    1. 量化模型
    2. 运行 lm-eval
    3. 比较 perplexity 与基线
    """
    # Quantize
    oneshot(model=model, recipe=config["recipe"])
    
    # Evaluate
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={save_dir}",
        tasks=["wikitext"],
    )
    
    # Compare
    perplexity = results["results"]["wikitext"]["perplexity"]
    assert perplexity < config["max_perplexity"]
```

---

### 7. test_timer/ - 性能计时测试

**用途**: 测量压缩操作的性能

```
test_timer/
├── __init__.py
├── timer.py                # 计时器
└── timer_utils.py          # 计时工具
```

**用法**:
```python
# timer.py
class CompressionTimer:
    """
    测量压缩各阶段的时间:
    - 模型加载时间
    - 校准时间
    - 量化时间
    - 保存时间
    """
    
    def time_oneshot(self, model, recipe, dataset):
        with Timer("total"):
            with Timer("load"):
                model = load_model()
            
            with Timer("calibrate"):
                calibration_dataloader = prepare_data()
            
            with Timer("quantize"):
                oneshot(model, recipe, dataset)
            
            with Timer("save"):
                model.save_pretrained()
        
        return self.timings
```

---

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定目录
```bash
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/llmcompressor/

# E2E 测试 (需要 GPU)
pytest tests/e2e/ -v
```

### 运行特定测试
```bash
# 按名称匹配
pytest tests/ -k "gptq"

# 运行特定文件
pytest tests/llmcompressor/modifiers/quantization/test_base.py

# 运行特定测试类
pytest tests/llmcompressor/modifiers/awq/test_base.py::TestAWQModifier
```

### 测试标记
```bash
# 跳过 GPU 测试
pytest tests/ -m "not gpu"

# 只运行快速测试
pytest tests/ -m "fast"

# 运行慢速测试
pytest tests/ -m "slow"
```

---

## 测试覆盖率

### 生成覆盖率报告
```bash
pytest tests/ --cov=llmcompressor --cov-report=html
```

### 查看报告
```bash
open htmlcov/index.html
```

---

## 添加新测试

### 单元测试模板
```python
# tests/unit/test_my_feature.py
import pytest
from llmcompressor.my_module import MyClass

class TestMyClass:
    @pytest.fixture
    def instance(self):
        return MyClass()
    
    def test_basic_functionality(self, instance):
        """测试基本功能"""
        result = instance.do_something()
        assert result is not None
    
    def test_edge_case(self, instance):
        """测试边界情况"""
        with pytest.raises(ValueError):
            instance.do_something_invalid()
```

### 集成测试模板
```python
# tests/llmcompressor/test_my_integration.py
import pytest
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot

class TestMyIntegration:
    @pytest.fixture
    def model(self):
        return AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype="auto",
        )
    
    def test_full_workflow(self, model, tmp_path):
        """测试完整工作流"""
        # Quantize
        oneshot(model=model, recipe=my_recipe)
        
        # Save
        model.save_pretrained(tmp_path)
        
        # Reload and verify
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
        assert reloaded is not None
```

---

## 测试配置文件

### conftest.py
```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def small_model():
    """加载小模型用于测试"""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

@pytest.fixture
def temp_output_dir(tmp_path):
    """提供临时输出目录"""
    return tmp_path / "output"
```

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    gpu: requires GPU
    slow: slow tests
    fast: fast tests
    
addopts = -v --tb=short
```

