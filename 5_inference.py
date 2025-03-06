"""
Inference Script for Phi-3 Vision (Step 5)

This script runs inference with the fine-tuned Phi-3 Vision model (PyTorch format).
It allows testing the model on new images with custom prompts after the model has been fine-tuned,
converted to ONNX, and evaluated.

Usage:
    python 5_inference.py --model_path ./fine_tuned_model --image_path ./processed_data/images/sample1.jpg --prompt "Describe what you see in this image."
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Phi-3 Vision model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./build/fine_tuned_model",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image for inference"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what you see in this image.",
        help="Text prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load components
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Determine if we're loading a base model or adapter
    has_adapter = False
    try:
        # Check if there's a lora_adapter directory
        lora_path = f"{args.model_path}/lora_adapter"
        import os
        if os.path.exists(lora_path):
            has_adapter = True
            print(f"Found LoRA adapter at {lora_path}")
    except:
        pass
    
    # Configure model loading
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    
    if torch.cuda.is_available():
        model_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
        })
    
    if has_adapter:
        # Load the base model
        base_model_name = "microsoft/Phi-3-vision-128k-instruct"
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        
        # Load the adapter
        model = PeftModel.from_pretrained(model, f"{args.model_path}/lora_adapter")
    else:
        # Load the full fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    # Load and preprocess the image
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    
    # Prepare the prompt with <image> tag for Phi-3 Vision
    prompt = args.prompt
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    
    # Format for chat-style inference
    chat_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
    
    # Process inputs
    inputs = processor(
        text=chat_prompt,
        images=image,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as the model
    for k, v in inputs.items():
        if hasattr(v, "to") and callable(v.to):
            inputs[k] = v.to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    try:
        assistant_response = generated_text.split("<|assistant|>")[1].strip()
    except:
        assistant_response = generated_text
    
    print("\n--- Generated Response ---")
    print(assistant_response)
    print("-------------------------")

if __name__ == "__main__":
    main()