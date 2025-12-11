# Directory Guide: examples/

这个文档详细解释 `llm-compressor/examples/` 目录的结构和每个示例的用途。

---

## 目录概览

```
examples/
├── autoround/                      # AutoRound 量化算法示例
├── awq/                            # AWQ 量化算法示例
├── big_models_with_sequential_onloading/  # 大模型顺序加载示例
├── compressed_inference/           # 压缩推理示例
├── finetuning/                     # 微调配置示例
├── model_free_ptq/                 # 无需模型的 PTQ 示例
├── multimodal_audio/               # 多模态音频模型示例
├── multimodal_vision/              # 多模态视觉模型示例
├── quantization_2of4_sparse_w4a16/ # 2:4 稀疏 + W4A16 量化
├── quantization_kv_cache/          # KV Cache 量化示例
├── quantization_non_uniform/       # 非均匀量化示例
├── quantization_w4a16/             # W4A16 (INT4 权重) 量化
├── quantization_w4a16_fp4/         # W4A16 FP4 格式
├── quantization_w4a4_fp4/          # NVFP4 (W4A4) 量化
├── quantization_w8a8_fp8/          # FP8 量化
├── quantization_w8a8_int8/         # INT8 量化
├── quantizing_moe/                 # MoE 模型量化
├── sparse_2of4_quantization_fp8/   # 2:4 稀疏 + FP8
└── transform/                      # 变换 (QuIP, SpinQuant)
```

---

## 详细说明

### 1. autoround/ - AutoRound 量化算法

**用途**: 使用 AutoRound 算法进行权重量化

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_example.py` | Llama-3 模型的 AutoRound 量化示例 |
| `README.md` | AutoRound 算法说明文档 |

**AutoRound 特点**:
- 使用 sign-gradient descent 优化 rounding 和 clipping 范围
- 结合了 PTQ 的效率和参数调优的适应性
- 适合追求高精度恢复的场景

**示例代码模式**:
```python
from llmcompressor.modifiers.autoround import AutoRoundModifier

recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    # AutoRound specific parameters
)
oneshot(model=model, recipe=recipe, dataset=calibration_data)
```

---

### 2. awq/ - AWQ 量化算法

**用途**: Activation-Weighted Quantization 示例

**文件**:
| 文件 | 说明 |
|------|------|
| `llama_example.py` | Llama 模型 AWQ 量化 |
| `qwen3_coder_moe_example.py` | Qwen3 Coder MoE 模型 |
| `qwen3_moe_example.py` | Qwen3 MoE 模型 |
| `qwen3-vl-30b-a3b-Instruct-example.py` | Qwen3 VL 多模态模型 |
| `README.md` | AWQ 算法说明 |

**AWQ 特点**:
- 通过观测激活值识别 "salient channels"
- 使用 scaling 保护重要通道
- 特别适合 W4A16 (4-bit 权重) 量化

**示例代码模式**:
```python
from llmcompressor.modifiers.awq import AWQModifier

recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    n_grid=20,           # grid search points
    duo_scaling=True,    # use both activation and weight for scaling
)
```

---

### 3. big_models_with_sequential_onloading/ - 大模型处理

**用途**: 演示如何处理无法完全放入 GPU 的大模型

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3.3_70b.py` | Llama-3.3 70B 模型量化示例 |
| `README.md` | Sequential Onloading 说明 |
| `assets/sequential_onloading.png` | 架构示意图 |

**Sequential Onloading 特点**:
- 逐层将模型加载到 GPU
- 量化后立即卸载，为下一层腾出空间
- 允许在有限显存上处理超大模型

**关键配置**:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",        # 自动分配设备
    torch_dtype=torch.float16,
    # 配合 accelerate 的 offloading
)
```

---

### 4. compressed_inference/ - 压缩推理

**用途**: 演示如何使用压缩后的模型进行推理

**文件**:
| 文件 | 说明 |
|------|------|
| `fp8_compressed_inference.py` | FP8 压缩模型推理示例 |

**特点**:
- 展示从加载到生成的完整流程
- 验证压缩模型的输出质量
- 性能基准测试

---

### 5. finetuning/ - 微调配置

**用途**: 量化感知微调 (QAT) 的配置示例

**文件**:
| 文件 | 说明 |
|------|------|
| `configure_fsdp.md` | FSDP 配置指南 |
| `example_alternating_recipe.yaml` | 交替训练 Recipe |
| `example_fsdp_config.yaml` | FSDP 配置文件 |
| `example_single_gpu_config.yaml` | 单 GPU 配置文件 |

**使用场景**:
- 需要 QAT (量化感知训练) 而非 PTQ
- 分布式训练配置
- 多阶段训练 Recipe

---

### 6. model_free_ptq/ - 无需加载模型的 PTQ

**用途**: Data-Free 量化方案，不需要校准数据

**文件**:
| 文件 | 说明 |
|------|------|
| `kimi_k2_thinking_fp8_block.py` | Kimi K2 FP8 Block 量化 |
| `kimi_k2_thinking_nvfp4a16.py` | Kimi K2 NVFP4 量化 |
| `README.md` | Model-Free PTQ 说明 |

**特点**:
- 不需要将完整模型加载到 GPU
- 适合超大模型 (如 DeepSeek V3 风格)
- 使用 Block-wise 量化策略

---

### 7. multimodal_audio/ - 多模态音频模型

**用途**: 语音/音频模型的量化

**文件**:
| 文件 | 说明 |
|------|------|
| `whisper_example.py` | Whisper 语音识别模型量化 |
| `README.md` | 音频模型量化说明 |

**支持的模型**:
- OpenAI Whisper (语音识别)
- 其他音频-语言模型

---

### 8. multimodal_vision/ - 多模态视觉模型

**用途**: 视觉-语言模型的量化

**文件**:
| 文件 | 说明 |
|------|------|
| `llama4_example.py` | Llama 4 多模态模型 |
| `llava_example.py` | LLaVA 模型 |
| `mllama_example.py` | MLlama 模型 |
| `gemma3_example.py` | Gemma 3 视觉模型 |
| `idefics3_example.py` | Idefics3 模型 |
| `internvl3_example.py` | InternVL3 模型 |
| `mistral3_example.py` | Mistral 3 视觉模型 |
| `phi3_vision_example.py` | Phi-3 Vision 模型 |
| `pixtral_example.py` | Pixtral 模型 |
| `qwen_2_5_vl_example.py` | Qwen 2.5 VL 模型 |
| `qwen2_vl_example.py` | Qwen 2 VL 模型 |
| `README.md`, `README_internvl3.md` | 说明文档 |

**特点**:
- 需要特殊处理视觉编码器
- 通常只量化语言模型部分
- 需要视觉数据作为校准集

---

### 9. quantization_2of4_sparse_w4a16/ - 稀疏 + 量化组合

**用途**: 2:4 Semi-structured Sparsity 与 W4A16 量化的组合

**文件**:
| 文件 | 说明 |
|------|------|
| `llama7b_sparse_w4a16.py` | Llama 7B 稀疏量化示例 |
| `2of4_w4a16_recipe.yaml` | 基础 Recipe |
| `2of4_w4a16_group-128_recipe.yaml` | Group-128 Recipe |
| `README.md` | 说明文档 |

**2:4 Sparsity 原理**:
```
原始权重: [a, b, c, d, e, f, g, h]
2:4 稀疏: [a, 0, c, 0, e, 0, g, 0]  # 每 4 个中保留 2 个

压缩效果: 2x (稀疏) × 4x (INT4) = 8x 总压缩
```

---

### 10. quantization_kv_cache/ - KV Cache 量化

**用途**: 量化 Key-Value Cache 以减少推理时内存

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_fp8_kv_example.py` | Llama 3 KV Cache FP8 量化 |
| `gemma2_fp8_kv_example.py` | Gemma 2 KV Cache 量化 |
| `phi3.5_fp8_kv_example.py` | Phi 3.5 KV Cache 量化 |
| `README.md` | 说明文档 |

**KV Cache 量化意义**:
- 长上下文推理时 KV Cache 占用大量内存
- 量化 KV Cache 可以支持更长的序列
- 通常使用 FP8 以平衡精度和压缩

**配置示例**:
```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="token",
    ),
)
```

---

### 11. quantization_non_uniform/ - 非均匀量化

**用途**: 对模型不同部分使用不同的量化精度

**文件**:
| 文件 | 说明 |
|------|------|
| `quantization_fp8_multiple_strategies.py` | FP8 多策略示例 |
| `quantization_int4_int8.py` | INT4 + INT8 混合 |
| `quantization_multiple_modifiers.py` | 多 Modifier 组合 |
| `quantization_nvfp4_fp8.py` | NVFP4 + FP8 混合 |
| `README.md` | 说明文档 |

**典型场景**:
```python
# MLP 使用 INT4, Attention 使用 INT8
recipe = [
    GPTQModifier(targets=mlp_layers, scheme="W4A16"),
    GPTQModifier(targets=attention_layers, scheme="W8A16"),
]
```

---

### 12. quantization_w4a16/ - INT4 权重量化

**用途**: 最常用的 4-bit 权重量化方案

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_example.py` | Llama 3 W4A16 GPTQ 量化 |
| `README.md` | 说明文档 |

**W4A16 含义**:
- **W4**: 权重量化到 4-bit
- **A16**: 激活保持 16-bit
- 压缩比: ~4x
- 通常使用 GPTQ 或 AWQ 算法

---

### 13. quantization_w4a16_fp4/ - FP4 权重量化

**用途**: 使用 FP4 (4-bit 浮点) 而非 INT4

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_example.py` | Llama 3 FP4 量化 |
| `qwen3_example.py` | Qwen 3 FP4 量化 |

**FP4 vs INT4**:
- FP4 有更好的动态范围表示
- 适合分布不均匀的权重
- 需要特定硬件支持

---

### 14. quantization_w4a4_fp4/ - NVFP4 (权重+激活都是 4-bit)

**用途**: NVIDIA Blackwell 专用的 FP4 格式

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_example.py` | Llama 3 NVFP4 |
| `llama4_example.py` | Llama 4 NVFP4 |
| `qwen_30b_a3b.py` | Qwen 30B MoE NVFP4 |
| `qwen3_next_example.py` | Qwen3 Next NVFP4 |
| `qwen3_vl_moe_w4a4_fp4.py` | Qwen3 VL MoE NVFP4 |
| `README.md` | 说明文档 |

**NVFP4 特点**:
- 权重和激活都是 4-bit
- 使用 Microscaling 技术
- 需要 NVIDIA Blackwell GPU

---

### 15. quantization_w8a8_fp8/ - FP8 量化

**用途**: 8-bit 浮点量化，适合 Hopper+ GPU

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_example.py` | 基础 FP8 量化 |
| `fp8_block_example.py` | Block-wise FP8 |
| `gemma2_example.py` | Gemma 2 FP8 |
| `granite4_example.py` | Granite 4 FP8 |
| `llama3.2_vision_example.py` | Llama 3.2 Vision |
| `llama4_fp8_block_example.py` | Llama 4 Block FP8 |
| `llava1.5_example.py` | LLaVA 1.5 |
| `qwen_2_5_vl_example.py` | Qwen 2.5 VL |
| `qwen2vl_example.py` | Qwen2 VL |
| `qwen3_next_example.py` | Qwen3 Next |
| `qwen3_vl_moe_fp8_example.py` | Qwen3 VL MoE |
| `whisper_example.py` | Whisper |
| `README.md`, `README_granite4.md` | 说明文档 |

**FP8 方案对比**:
| 方案 | 校准数据 | 特点 |
|------|----------|------|
| FP8_DYNAMIC | 不需要 | 最简单，激活动态量化 |
| FP8_STATIC | 需要 | 更高效推理 |
| FP8_BLOCK | 不需要 | Block-wise 量化，类似 DeepSeek V3 |

---

### 16. quantization_w8a8_int8/ - INT8 量化

**用途**: 传统 INT8 量化，广泛硬件支持

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_example.py` | Llama 3 INT8 |
| `gemma2_example.py` | Gemma 2 INT8 |
| `README.md` | 说明文档 |

**INT8 特点**:
- 支持所有 GPU 和 CPU
- 通常需要 SmoothQuant 预处理
- 比 FP8 更广泛的硬件兼容性

---

### 17. quantizing_moe/ - MoE 模型量化

**用途**: Mixture-of-Experts 模型的量化

**文件**:
| 文件 | 说明 |
|------|------|
| `mixtral_example.py` | Mixtral 8x7B |
| `qwen_example.py` | Qwen MoE |
| `deepseek_r1_example.py` | DeepSeek R1 |
| `README.md` | 说明文档 |

**MoE 量化注意事项**:
- 需要校准所有 experts
- 使用 `moe_calibrate_all_experts=True`
- Router 层通常不量化

---

### 18. sparse_2of4_quantization_fp8/ - 稀疏 + FP8

**用途**: 2:4 稀疏与 FP8 量化组合

**文件**:
| 文件 | 说明 |
|------|------|
| `llama3_8b_2of4.py` | Llama 3 8B 稀疏 FP8 |
| `README.md` | 说明文档 |

**组合优势**:
- 稀疏: 2x 压缩 + 计算加速
- FP8: 2x 压缩
- 总计: ~4x 压缩

---

### 19. transform/ - 权重变换

**用途**: 应用变换以提高量化精度

**文件**:
| 文件 | 说明 |
|------|------|
| `quip_example.py` | QuIP (Hadamard) 变换 |
| `spinquant_example.py` | SpinQuant 变换 |
| `README.md` | 说明文档 |

**变换原理**:
- 在权重和激活之间插入正交变换
- 使权重分布更均匀，减少量化误差
- 推理时需要应用逆变换

---

## 示例选择指南

### 按场景选择

| 场景 | 推荐示例 | 原因 |
|------|----------|------|
| 快速开始 | `quantization_w8a8_fp8/llama3_example.py` | 简单，无需校准数据 |
| 最高压缩 | `quantization_w4a16/llama3_example.py` | 4x 压缩 |
| 最佳精度 | `awq/llama_example.py` | AWQ 精度恢复好 |
| 超大模型 | `big_models_with_sequential_onloading/` | 内存高效 |
| 多模态 | `multimodal_vision/llava_example.py` | 视觉模型示例 |
| MoE 模型 | `quantizing_moe/mixtral_example.py` | MoE 专用处理 |

### 按硬件选择

| GPU | 推荐方案 |
|-----|----------|
| H100/Hopper+ | FP8 Dynamic, NVFP4 |
| A100/Ampere | INT8, W4A16 (GPTQ) |
| RTX 4090 | W4A16 (GPTQ/AWQ) |
| 旧款 GPU | INT8, W4A16 |
| CPU | INT8 |

