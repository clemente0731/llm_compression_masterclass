"""
================================================================================
Exercise 1: Hello World of Compression
================================================================================

Learning Objectives:
1. Understand the basic oneshot() API
2. Understand QuantizationModifier configuration
3. Understand FP8 Dynamic quantization scheme

Key Concepts:
- FP8 Dynamic: Weights are quantized to FP8, activations are dynamically 
  quantized per-token during inference
- No calibration data needed for FP8 Dynamic
- This is the simplest starting point for model compression

Expected Output:
- A quantized model saved to disk
- Sample generation to verify the model works
================================================================================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def run_hello_world_quantization():
    """
    Step-by-step FP8 Dynamic Quantization.
    
    This exercise demonstrates the simplest possible quantization workflow.
    """
    
    print("=" * 70)
    print("Exercise 1: Hello World of Compression")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load Model and Tokenizer
    # =========================================================================
    # We use TinyLlama because it's small (~1.1B parameters) and downloads quickly.
    # In production, you would use a larger model like Llama-3-8B.
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\n[Step 1] Loading model: {model_id}")
    print("This may take a moment if downloading for the first time...")
    
    try:
        # device_map="auto" automatically distributes model across available devices
        # torch_dtype="auto" uses the model's default precision (usually float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Ensure pad_token is set (required for batched inference)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Model loaded successfully!")
        print(f"Model dtype: {model.dtype}")
        print(f"Model device: {model.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: Make sure you have enough disk space and internet connection.")
        return
    
    # =========================================================================
    # STEP 2: Define the Quantization Recipe
    # =========================================================================
    # A Recipe is a declarative description of what optimizations to apply.
    # QuantizationModifier is the simplest modifier - it just applies PTQ.
    
    print("\n[Step 2] Configuring Quantization Recipe")
    
    recipe = QuantizationModifier(
        # targets: Which layers to quantize
        # "Linear" matches all torch.nn.Linear modules
        # You can also use regex patterns like "model.layers.\\d+.mlp.*"
        targets="Linear",
        
        # scheme: The quantization scheme to use
        # "FP8_DYNAMIC" means:
        #   - Weights: FP8 (8-bit floating point), static per-channel
        #   - Activations: FP8, dynamic per-token (computed at runtime)
        # This scheme does NOT require calibration data!
        scheme="FP8_DYNAMIC",
        
        # ignore: Layers to skip quantization
        # lm_head is the output layer that maps hidden states to vocabulary
        # It's very sensitive to quantization and should usually be kept in FP16
        ignore=["lm_head"],
    )
    
    print(f"Recipe configured:")
    print(f"  - Targets: Linear layers")
    print(f"  - Scheme: FP8_DYNAMIC")
    print(f"  - Ignored: lm_head")
    
    # =========================================================================
    # STEP 3: Apply Quantization
    # =========================================================================
    # oneshot() is the main entry point for compression.
    # For FP8_DYNAMIC, no calibration data is needed, so we don't pass dataset.
    
    print("\n[Step 3] Applying Quantization")
    print("This may take a few minutes...")
    
    try:
        # For FP8_DYNAMIC, we don't need calibration data
        # The oneshot function will automatically select DataFreePipeline
        oneshot(
            model=model,
            recipe=recipe,
            # No dataset needed for FP8_DYNAMIC!
        )
        print("Quantization completed successfully!")
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        return
    
    # =========================================================================
    # STEP 4: Verify the Model Works
    # =========================================================================
    # Always generate some samples to verify the quantized model produces
    # sensible output.
    
    print("\n[Step 4] Verifying Model Output")
    
    try:
        # Prepare input
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # =========================================================================
    # STEP 5: Save the Quantized Model
    # =========================================================================
    # The model is saved in safetensors format with quantization metadata.
    # This can be directly loaded by vLLM for inference.
    
    print("\n[Step 5] Saving Quantized Model")
    
    output_dir = "./tinyllama-fp8-dynamic"
    
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")
        
        # Print file sizes for comparison
        import os
        total_size = 0
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                total_size += size
                print(f"  {f}: {size / 1024 / 1024:.2f} MB")
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("Exercise 1 Complete!")
    print("=" * 70)
    print("""
Key Takeaways:
1. QuantizationModifier is the simplest way to quantize a model
2. FP8_DYNAMIC doesn't need calibration data
3. Always ignore lm_head for better accuracy
4. Always verify output after quantization
5. Models are saved in safetensors format for vLLM compatibility

Next Steps:
- Try Exercise 2 to learn about GPTQ with calibration data
- Try Exercise 3 to inspect the quantized weights
""")


if __name__ == "__main__":
    run_hello_world_quantization()
