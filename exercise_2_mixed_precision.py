"""
================================================================================
Exercise 2: The Sniper Approach (Mixed Precision Quantization)
================================================================================

Learning Objectives:
1. Understand GPTQModifier and how it differs from QuantizationModifier
2. Learn how to target specific layers using regex patterns
3. Understand the GPTQ algorithm at a high level
4. Learn how to prepare and use calibration data

Key Concepts:
- GPTQ uses Hessian information for intelligent weight quantization
- Calibration data is required to compute the Hessian matrix
- Different layers can be quantized with different precision (mixed precision)
- MLP layers are often more tolerant to quantization than Attention layers

Expected Output:
- A mixed-precision quantized model
- Understanding of how to control quantization granularity
================================================================================
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


def prepare_calibration_dataset(tokenizer, num_samples=256, max_length=1024):
    """
    Prepare a calibration dataset for GPTQ.
    
    The calibration dataset is used to:
    1. Compute the Hessian matrix for each layer
    2. Determine the sensitivity of each weight
    
    Best Practices:
    - Use data similar to your deployment scenario
    - 256-512 samples is usually sufficient
    - Longer sequences capture more patterns
    
    Args:
        tokenizer: The tokenizer for the model
        num_samples: Number of calibration samples
        max_length: Maximum sequence length
    
    Returns:
        A tokenized dataset ready for calibration
    """
    print(f"\n[Preparing Calibration Data]")
    print(f"  Samples: {num_samples}")
    print(f"  Max Length: {max_length}")
    
    # Load a subset of the dataset
    # open_platypus is a good general-purpose dataset for chat models
    try:
        ds = load_dataset(
            "garage-bAInd/Open-Platypus",
            split=f"train[:{num_samples}]",
        )
    except Exception as e:
        print(f"Error loading Open-Platypus: {e}")
        print("Falling back to built-in dataset name...")
        # oneshot() can handle dataset names directly
        return "open_platypus"
    
    # Preprocess: combine instruction and output
    def preprocess(example):
        text = f"### Instruction:\n{example['instruction']}\n\n"
        text += f"### Response:\n{example['output']}"
        return {"text": text}
    
    ds = ds.map(preprocess)
    
    # Tokenize
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )
    
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    
    print(f"  Dataset prepared with {len(ds)} samples")
    return ds


def create_mlp_only_recipe():
    """
    Create a recipe that only quantizes MLP layers.
    
    Why target only MLP?
    - MLP layers make up ~2/3 of model parameters
    - MLP is generally more tolerant to quantization
    - Keeping Attention in higher precision preserves reasoning capability
    
    This is a common strategy for balancing compression and accuracy.
    """
    
    print("\n[Creating MLP-Only Recipe]")
    
    # Regex patterns for Llama-style MLP layers
    # These patterns match:
    #   - model.layers.0.mlp.gate_proj
    #   - model.layers.0.mlp.up_proj
    #   - model.layers.0.mlp.down_proj
    #   - model.layers.1.mlp.gate_proj
    #   - ... and so on
    
    mlp_targets = [
        "model.layers.\\d+.mlp.gate_proj",
        "model.layers.\\d+.mlp.up_proj",
        "model.layers.\\d+.mlp.down_proj",
    ]
    
    print(f"  Targeting layers:")
    for t in mlp_targets:
        print(f"    - {t}")
    
    recipe = GPTQModifier(
        # targets: Use regex to match specific layers
        targets=mlp_targets,
        
        # scheme: W4A16 means 4-bit weights, 16-bit activations
        # This gives ~4x compression on weights
        scheme="W4A16",
        
        # ignore: Always ignore lm_head
        ignore=["lm_head"],
        
        # GPTQ-specific parameters:
        
        # block_size: Number of columns to process at once
        # Larger = faster but uses more memory
        block_size=128,
        
        # dampening_frac: Damping factor for Hessian stability
        # Increase if you see numerical issues (NaN, Inf)
        dampening_frac=0.01,
        
        # actorder: Activation ordering strategy
        # "static" is recommended (best accuracy, no runtime cost)
        actorder="static",
    )
    
    print(f"  Scheme: W4A16 (4-bit weights, 16-bit activations)")
    print(f"  Block size: 128")
    print(f"  Dampening: 0.01")
    
    return recipe


def create_attention_only_recipe():
    """
    Create a recipe that only quantizes Attention layers (for comparison).
    
    Attention layers are generally MORE sensitive to quantization because:
    1. Q/K dot product is sensitive to small changes
    2. Attention patterns are critical for understanding context
    3. Errors in attention propagate through the entire sequence
    
    We use INT8 (higher precision) for attention layers.
    """
    
    print("\n[Creating Attention-Only Recipe]")
    
    attention_targets = [
        "model.layers.\\d+.self_attn.q_proj",
        "model.layers.\\d+.self_attn.k_proj",
        "model.layers.\\d+.self_attn.v_proj",
        "model.layers.\\d+.self_attn.o_proj",
    ]
    
    print(f"  Targeting layers:")
    for t in attention_targets:
        print(f"    - {t}")
    
    recipe = GPTQModifier(
        targets=attention_targets,
        # Use INT8 for attention (more precision)
        scheme="W8A16",
        ignore=["lm_head"],
        block_size=128,
        dampening_frac=0.01,
    )
    
    print(f"  Scheme: W8A16 (8-bit weights, 16-bit activations)")
    
    return recipe


def run_mixed_precision_quantization():
    """
    Main function: Apply mixed-precision quantization.
    """
    
    print("=" * 70)
    print("Exercise 2: Mixed Precision Quantization")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load Model
    # =========================================================================
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\n[Step 1] Loading model: {model_id}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # =========================================================================
    # STEP 2: Prepare Calibration Data
    # =========================================================================
    
    print("\n[Step 2] Preparing Calibration Data")
    
    # For GPTQ, we NEED calibration data to compute Hessian
    # Unlike FP8_DYNAMIC, this is not optional!
    
    calibration_data = prepare_calibration_dataset(
        tokenizer,
        num_samples=256,  # 256 is usually enough
        max_length=1024,  # Longer sequences are better
    )
    
    # =========================================================================
    # STEP 3: Create Recipe
    # =========================================================================
    
    print("\n[Step 3] Creating Quantization Recipe")
    
    # Option A: Only quantize MLP layers (more aggressive)
    recipe = create_mlp_only_recipe()
    
    # Option B: Only quantize Attention layers (for comparison)
    # recipe = create_attention_only_recipe()
    
    # Option C: Quantize both with different precision (true mixed precision)
    # recipe = [
    #     create_mlp_only_recipe(),      # MLP: W4A16
    #     create_attention_only_recipe(), # Attention: W8A16
    # ]
    
    # =========================================================================
    # STEP 4: Apply Quantization
    # =========================================================================
    
    print("\n[Step 4] Applying GPTQ Quantization")
    print("This may take several minutes (GPTQ is more compute-intensive)...")
    
    try:
        oneshot(
            model=model,
            recipe=recipe,
            dataset=calibration_data,
            num_calibration_samples=256,
            max_seq_length=1024,
        )
        print("Quantization completed successfully!")
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # STEP 5: Verify Output
    # =========================================================================
    
    print("\n[Step 5] Verifying Model Output")
    
    try:
        prompt = "What is machine learning?"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated:\n{generated}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
    
    # =========================================================================
    # STEP 6: Save Model
    # =========================================================================
    
    print("\n[Step 6] Saving Model")
    
    output_dir = "./tinyllama-w4a16-mlp-only"
    
    try:
        model.save_pretrained(output_dir, save_compressed=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Exercise 2 Complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. GPTQModifier requires calibration data (unlike QuantizationModifier + FP8)
2. Use regex patterns in 'targets' to precisely control which layers to quantize
3. MLP layers are more tolerant to aggressive quantization (INT4)
4. Attention layers should use higher precision (INT8) for better accuracy
5. Mixed precision strategies balance compression ratio and model quality

GPTQ Algorithm Summary:
1. Forward pass calibration data to collect Hessian (H = X @ X.T)
2. For each weight column:
   a. Quantize the column
   b. Propagate error to remaining columns using H^{-1}
3. This "error compensation" preserves output quality

Next Steps:
- Try Exercise 3 to inspect the quantized weights
- Experiment with different group_size values
- Try combining MLP and Attention quantization
""")


if __name__ == "__main__":
    run_mixed_precision_quantization()
