"""
================================================================================
Exercise 3: The Researcher's Probe (Model Inspection)
================================================================================

Learning Objectives:
1. Learn how to inspect quantized model weights
2. Understand the structure of a quantized model
3. Verify that quantization was actually applied
4. Understand scale, zero_point, and g_idx parameters

Key Concepts:
- Quantized models have additional parameters: weight_scale, weight_zero_point
- Group-wise quantization also has weight_g_idx (group index mapping)
- Different quantization schemes result in different weight dtypes
- Always verify quantization before deploying to production

Expected Output:
- Detailed inspection report of model quantization status
- Statistics about quantized vs non-quantized layers
================================================================================
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def print_separator(title=""):
    """Print a formatted separator."""
    width = 70
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def get_dtype_name(dtype):
    """Get human-readable dtype name."""
    dtype_names = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
        torch.int8: "INT8",
        torch.int16: "INT16",
        torch.int32: "INT32",
        torch.uint8: "UINT8",
    }
    
    # Handle FP8 types (may not exist in older PyTorch)
    dtype_str = str(dtype)
    if "float8_e4m3fn" in dtype_str:
        return "FP8_E4M3"
    if "float8_e5m2" in dtype_str:
        return "FP8_E5M2"
    
    return dtype_names.get(dtype, str(dtype))


def inspect_module_quantization(name, module):
    """
    Inspect a single module for quantization artifacts.
    
    Returns:
        dict with quantization info, or None if not quantized
    """
    info = {
        "name": name,
        "type": module.__class__.__name__,
        "is_quantized": False,
        "weight_dtype": None,
        "weight_shape": None,
        "has_scale": False,
        "has_zero_point": False,
        "has_g_idx": False,
        "scale_shape": None,
        "quantization_scheme": None,
    }
    
    # Check if module has weight
    if not hasattr(module, "weight"):
        return None
    
    weight = module.weight
    info["weight_dtype"] = get_dtype_name(weight.dtype)
    info["weight_shape"] = list(weight.shape)
    
    # Check for quantization artifacts
    # Method 1: Check for quantization-specific attributes
    quantization_attrs = [
        "weight_scale",
        "weight_zero_point",
        "weight_g_idx",
        "quantization_scheme",
    ]
    
    for attr in quantization_attrs:
        if hasattr(module, attr):
            val = getattr(module, attr)
            if val is not None:
                info["is_quantized"] = True
                
                if attr == "weight_scale":
                    info["has_scale"] = True
                    info["scale_shape"] = list(val.shape) if hasattr(val, "shape") else None
                elif attr == "weight_zero_point":
                    info["has_zero_point"] = True
                elif attr == "weight_g_idx":
                    info["has_g_idx"] = True
                elif attr == "quantization_scheme":
                    info["quantization_scheme"] = str(val)
    
    # Method 2: Check dtype (some quantized models use int8/fp8 directly)
    if weight.dtype in [torch.int8, torch.uint8]:
        info["is_quantized"] = True
    
    # Check for FP8 types
    dtype_str = str(weight.dtype)
    if "float8" in dtype_str:
        info["is_quantized"] = True
    
    return info


def analyze_quantization_config(model_path):
    """
    Analyze the quantization configuration from config.json.
    """
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        return None
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Look for quantization-related fields
    quant_info = {}
    
    # Check for compressed_tensors config
    if "quantization_config" in config:
        quant_info["quantization_config"] = config["quantization_config"]
    
    # Check for specific quantization libraries
    for key in ["quant_method", "bits", "group_size", "desc_act"]:
        if key in config:
            quant_info[key] = config[key]
    
    return quant_info if quant_info else None


def inspect_model_weights(model_path_or_obj):
    """
    Comprehensive inspection of model weights for quantization.
    
    Args:
        model_path_or_obj: Either a path to saved model or a model object
    """
    
    print_separator("Model Inspection Report")
    
    # =========================================================================
    # Load Model
    # =========================================================================
    
    if isinstance(model_path_or_obj, str):
        model_path = model_path_or_obj
        
        if not os.path.exists(model_path):
            print(f"\nError: Path '{model_path}' does not exist.")
            print("Tip: Run exercise_1 or exercise_2 first to generate a model.")
            return
        
        print(f"\nModel Path: {model_path}")
        
        # Check config first
        quant_config = analyze_quantization_config(model_path)
        if quant_config:
            print("\n[Quantization Config from config.json]")
            print(json.dumps(quant_config, indent=2))
        
        # Load model
        print("\nLoading model for inspection...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",  # Use CPU to avoid GPU memory issues
                torch_dtype="auto",
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        model = model_path_or_obj
        model_path = "In-Memory Model"
    
    # =========================================================================
    # Collect Module Information
    # =========================================================================
    
    print("\n[Scanning Modules...]")
    
    module_infos = []
    layer_type_counts = defaultdict(int)
    quantized_layer_type_counts = defaultdict(int)
    
    for name, module in model.named_modules():
        info = inspect_module_quantization(name, module)
        if info is not None:
            module_infos.append(info)
            layer_type_counts[info["type"]] += 1
            if info["is_quantized"]:
                quantized_layer_type_counts[info["type"]] += 1
    
    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    
    print_separator("Summary Statistics")
    
    total_layers = len(module_infos)
    quantized_layers = sum(1 for info in module_infos if info["is_quantized"])
    non_quantized_layers = total_layers - quantized_layers
    
    print(f"\nTotal Layers with Weights: {total_layers}")
    print(f"Quantized Layers: {quantized_layers}")
    print(f"Non-Quantized Layers: {non_quantized_layers}")
    
    if total_layers > 0:
        print(f"Quantization Coverage: {quantized_layers / total_layers * 100:.1f}%")
    
    # =========================================================================
    # Layer Type Breakdown
    # =========================================================================
    
    print_separator("Layer Type Breakdown")
    
    print(f"\n{'Layer Type':<30} {'Total':>10} {'Quantized':>10} {'Coverage':>10}")
    print("-" * 60)
    
    for layer_type, count in sorted(layer_type_counts.items()):
        quant_count = quantized_layer_type_counts[layer_type]
        coverage = quant_count / count * 100 if count > 0 else 0
        print(f"{layer_type:<30} {count:>10} {quant_count:>10} {coverage:>9.1f}%")
    
    # =========================================================================
    # Data Type Distribution
    # =========================================================================
    
    print_separator("Weight Data Types")
    
    dtype_counts = defaultdict(int)
    for info in module_infos:
        dtype_counts[info["weight_dtype"]] += 1
    
    print(f"\n{'Data Type':<20} {'Count':>10}")
    print("-" * 30)
    for dtype, count in sorted(dtype_counts.items()):
        print(f"{dtype:<20} {count:>10}")
    
    # =========================================================================
    # Quantization Parameters
    # =========================================================================
    
    print_separator("Quantization Parameters")
    
    has_scale_count = sum(1 for info in module_infos if info["has_scale"])
    has_zp_count = sum(1 for info in module_infos if info["has_zero_point"])
    has_gidx_count = sum(1 for info in module_infos if info["has_g_idx"])
    
    print(f"\nLayers with weight_scale: {has_scale_count}")
    print(f"Layers with weight_zero_point: {has_zp_count}")
    print(f"Layers with weight_g_idx: {has_gidx_count}")
    
    # =========================================================================
    # Sample Quantized Layers
    # =========================================================================
    
    print_separator("Sample Quantized Layers")
    
    quantized_infos = [info for info in module_infos if info["is_quantized"]]
    
    if not quantized_infos:
        print("\nNo quantized layers found!")
        print("This may indicate:")
        print("  1. The model was not quantized")
        print("  2. The quantization format is not recognized")
        print("  3. The model uses a different quantization scheme")
    else:
        # Show first 5 quantized layers
        print("\nFirst 5 quantized layers:")
        for i, info in enumerate(quantized_infos[:5]):
            print(f"\n  [{i+1}] {info['name']}")
            print(f"      Type: {info['type']}")
            print(f"      Weight dtype: {info['weight_dtype']}")
            print(f"      Weight shape: {info['weight_shape']}")
            if info['has_scale']:
                print(f"      Scale shape: {info['scale_shape']}")
            if info['has_g_idx']:
                print(f"      Has g_idx: Yes (group-wise quantization)")
    
    # =========================================================================
    # Sample Non-Quantized Layers
    # =========================================================================
    
    print_separator("Sample Non-Quantized Layers")
    
    non_quantized_infos = [info for info in module_infos if not info["is_quantized"]]
    
    if non_quantized_infos:
        print("\nFirst 5 non-quantized layers:")
        for i, info in enumerate(non_quantized_infos[:5]):
            print(f"\n  [{i+1}] {info['name']}")
            print(f"      Type: {info['type']}")
            print(f"      Weight dtype: {info['weight_dtype']}")
            print(f"      Weight shape: {info['weight_shape']}")
    else:
        print("\nAll layers are quantized!")
    
    # =========================================================================
    # Detailed Weight Statistics
    # =========================================================================
    
    print_separator("Detailed Weight Statistics")
    
    # Pick a sample quantized layer for detailed analysis
    sample_layer = None
    sample_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_scale"):
            sample_layer = module
            sample_name = name
            break
    
    if sample_layer is not None:
        print(f"\nDetailed analysis of: {sample_name}")
        
        weight = sample_layer.weight
        print(f"\n  Weight:")
        print(f"    Shape: {weight.shape}")
        print(f"    Dtype: {weight.dtype}")
        print(f"    Min: {weight.min().item():.6f}")
        print(f"    Max: {weight.max().item():.6f}")
        print(f"    Mean: {weight.float().mean().item():.6f}")
        print(f"    Std: {weight.float().std().item():.6f}")
        
        if hasattr(sample_layer, "weight_scale"):
            scale = sample_layer.weight_scale
            print(f"\n  Scale:")
            print(f"    Shape: {scale.shape}")
            print(f"    Dtype: {scale.dtype}")
            print(f"    Min: {scale.min().item():.6f}")
            print(f"    Max: {scale.max().item():.6f}")
        
        if hasattr(sample_layer, "weight_zero_point"):
            zp = sample_layer.weight_zero_point
            if zp is not None:
                print(f"\n  Zero Point:")
                print(f"    Shape: {zp.shape}")
                print(f"    Dtype: {zp.dtype}")
    else:
        print("\nNo quantized layer with scale found for detailed analysis.")
    
    # =========================================================================
    # Conclusion
    # =========================================================================
    
    print_separator("Conclusion")
    
    if quantized_layers == 0:
        print("\nResult: MODEL IS NOT QUANTIZED")
        print("The model appears to be in its original floating-point format.")
    elif quantized_layers == total_layers:
        print("\nResult: MODEL IS FULLY QUANTIZED")
        print("All layers with weights have been quantized.")
    else:
        print("\nResult: MODEL IS PARTIALLY QUANTIZED (Mixed Precision)")
        print(f"  - {quantized_layers} layers quantized")
        print(f"  - {non_quantized_layers} layers in original precision")
    
    print_separator()


def main():
    """Main entry point."""
    
    # Default paths to check
    default_paths = [
        "./tinyllama-fp8-dynamic",      # From Exercise 1
        "./tinyllama-w4a16-mlp-only",   # From Exercise 2
        "./tinyllama-w4a16-mixed",      # Alternative name
    ]
    
    if len(sys.argv) > 1:
        # Use provided path
        target_path = sys.argv[1]
    else:
        # Find first existing path
        target_path = None
        for path in default_paths:
            if os.path.exists(path):
                target_path = path
                break
        
        if target_path is None:
            print("No quantized model found!")
            print("\nExpected one of:")
            for path in default_paths:
                print(f"  - {path}")
            print("\nPlease run exercise_1 or exercise_2 first to generate a model,")
            print("or provide a path as argument:")
            print("  python exercise_3_inspection.py /path/to/model")
            return
    
    print(f"Inspecting: {target_path}")
    inspect_model_weights(target_path)
    
    print("""
================================================================================
Key Takeaways:
1. Quantized models have additional parameters: weight_scale, weight_zero_point
2. Group-wise quantization also has weight_g_idx
3. Different schemes result in different weight dtypes
4. Always verify quantization before production deployment
5. Check config.json for quantization metadata

Next Steps:
- Compare perplexity between original and quantized models
- Try running inference with vLLM on the quantized model
- Experiment with different quantization schemes
================================================================================
""")


if __name__ == "__main__":
    main()
