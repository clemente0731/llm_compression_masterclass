# Directory Guide: tools/

这个文档详细解释 `llm-compressor/tools/` 目录的工具脚本。

---

## 目录概览

```
tools/
└── collect_env.py          # 环境信息收集工具
```

目前 `tools/` 目录只包含一个工具脚本，用于收集系统环境信息以便调试和问题报告。

---

## collect_env.py - 环境信息收集

### 用途

当用户报告问题时，开发者需要了解用户的运行环境。这个脚本自动收集所有相关信息。

### 运行方式

```bash
# 从项目根目录运行
python tools/collect_env.py

# 或者从任意位置运行
python -m llmcompressor.tools.collect_env
```

### 收集的信息

#### 1. 系统信息
```
Platform: Linux-5.15.0-generic-x86_64-with-glibc2.31
Python: 3.10.12
```

#### 2. PyTorch 信息
```
PyTorch: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
cuDNN version: 8.9.0
GPU: NVIDIA A100-SXM4-80GB
Number of GPUs: 8
```

#### 3. 关键依赖版本
```
transformers: 4.36.0
accelerate: 0.25.0
safetensors: 0.4.0
compressed-tensors: 0.5.0
llmcompressor: 0.8.1
vllm: 0.2.7
```

#### 4. 环境变量
```
CUDA_HOME: /usr/local/cuda-12.1
CUDA_VISIBLE_DEVICES: 0,1,2,3
HF_HOME: /home/user/.cache/huggingface
```

### 源码解析

```python
# tools/collect_env.py

import sys
import platform
import subprocess

def get_system_info():
    """收集系统信息"""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

def get_pytorch_info():
    """收集 PyTorch 信息"""
    import torch
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
    
    return info

def get_package_versions():
    """收集关键包版本"""
    packages = [
        "transformers",
        "accelerate",
        "safetensors",
        "compressed-tensors",
        "llmcompressor",
        "vllm",
        "datasets",
        "tokenizers",
    ]
    
    versions = {}
    for pkg in packages:
        try:
            import importlib
            mod = importlib.import_module(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    
    return versions

def get_env_vars():
    """收集相关环境变量"""
    import os
    relevant_vars = [
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_TOKEN",
        "TRANSFORMERS_CACHE",
        "LLM_COMPRESSOR_LOG_FILE",
    ]
    
    return {
        var: os.environ.get(var, "not set")
        for var in relevant_vars
    }

def main():
    """主函数：收集并打印所有信息"""
    print("=" * 60)
    print("LLM Compressor Environment Information")
    print("=" * 60)
    
    print("\n[System Information]")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")
    
    print("\n[PyTorch Information]")
    for key, value in get_pytorch_info().items():
        print(f"  {key}: {value}")
    
    print("\n[Package Versions]")
    for pkg, version in get_package_versions().items():
        print(f"  {pkg}: {version}")
    
    print("\n[Environment Variables]")
    for var, value in get_env_vars().items():
        # 隐藏敏感信息
        if "TOKEN" in var and value != "not set":
            value = "***"
        print(f"  {var}: {value}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

### 输出示例

```
============================================================
LLM Compressor Environment Information
============================================================

[System Information]
  platform: Linux-5.15.0-generic-x86_64-with-glibc2.31
  python_version: 3.10.12 (main, Jun 11 2023, 05:26:28)
  python_executable: /home/user/miniconda3/envs/llm/bin/python

[PyTorch Information]
  pytorch_version: 2.1.0+cu121
  cuda_available: True
  cuda_version: 12.1
  cudnn_version: 8900
  gpu_count: 8
  gpu_names: ['NVIDIA A100-SXM4-80GB', 'NVIDIA A100-SXM4-80GB', ...]

[Package Versions]
  transformers: 4.36.0
  accelerate: 0.25.0
  safetensors: 0.4.0
  compressed-tensors: 0.5.0
  llmcompressor: 0.8.1
  vllm: 0.2.7
  datasets: 2.15.0
  tokenizers: 0.15.0

[Environment Variables]
  CUDA_HOME: /usr/local/cuda-12.1
  CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
  HF_HOME: /home/user/.cache/huggingface
  HF_TOKEN: ***
  TRANSFORMERS_CACHE: not set
  LLM_COMPRESSOR_LOG_FILE: not set

============================================================
```

---

## 在 Issue 中使用

### 提交 Bug 报告时

在提交 GitHub Issue 时，请附上环境信息：

```markdown
### Environment

<details>
<summary>Environment Info (click to expand)</summary>

```
[粘贴 python tools/collect_env.py 的输出]
```

</details>

### Problem Description

[描述问题]

### Steps to Reproduce

1. ...
2. ...

### Expected Behavior

[期望的行为]

### Actual Behavior

[实际的行为]
```

---

## 扩展工具

### 建议添加的工具

如果你想为项目贡献更多工具，以下是一些建议：

#### 1. 模型检查工具
```python
# tools/inspect_model.py
"""
检查量化模型的详细信息:
- 量化层数量
- Scale/Zero-point 分布
- 压缩率统计
"""
```

#### 2. 性能基准工具
```python
# tools/benchmark.py
"""
测量量化性能:
- 内存使用
- 推理速度
- 压缩时间
"""
```

#### 3. 精度评估工具
```python
# tools/evaluate.py
"""
评估量化模型精度:
- Perplexity
- 下游任务准确率
- 与原始模型对比
"""
```

#### 4. 模型转换工具
```python
# tools/convert.py
"""
在不同量化格式之间转换:
- GPTQ -> AWQ
- FP8 -> INT8
- 导出到其他框架
"""
```

---

## 开发工具建议

### 本地开发常用命令

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/llmcompressor

# 运行测试
pytest tests/ -v

# 生成文档
mkdocs serve
```

### 调试技巧

#### 启用详细日志
```bash
export LLM_COMPRESSOR_LOG_FILE=debug.log
python your_script.py
```

#### 使用 pdb 调试
```python
import pdb; pdb.set_trace()
```

#### 检查 GPU 内存
```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

---

## 相关资源

### 项目工具
- `tools/collect_env.py` - 环境信息收集

### 文档工具
- `mkdocs.yml` - MkDocs 文档配置
- `docs/` - 文档源文件

### CI/CD 工具
- `.github/workflows/` - GitHub Actions 配置
- `Makefile` - 常用命令封装

### 代码质量工具
- `pyproject.toml` - 项目配置
- `.pre-commit-config.yaml` - Pre-commit 钩子（如果有）

