"""
ONNX Conversion Script for Phi-3 Vision

This script converts a fine-tuned Phi-3 Vision model to ONNX format for deployment
with ONNX Runtime. It handles:
1. Loading the fine-tuned model
2. Creating the necessary dummy inputs
3. Exporting to ONNX format with optimizations
4. Saving the model and required configuration files

Usage:
    python 3_convert_to_onnx.py --model_dir ./build/fine_tuned_model --output_dir ./build/onnx_model
"""

import os
import json
import argparse
import logging
import torch
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert fine-tuned Phi-3 Vision model to ONNX format")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./build/fine_tuned_model",
        help="Directory containing the fine-tuned model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./build/onnx_model",
        help="Directory to save the ONNX model"
    )
        type=int,
        default=2048,
        help="Maximum sequence length for model inputs"
    )
    parser.add_argument(
        "--with_image_input",
        action="store_true",
        help="Include image input in the ONNX model"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ONNX Runtime optimizations"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the ONNX model for reduced size and faster inference"
    )
    
    return parser.parse_args()

def prepare_dummy_inputs(model, tokenizer, processor, args):
    """Prepare dummy inputs for ONNX export based on model architecture"""
    batch_size = 1
    seq_length = args.max_seq_length
    
    # Create dummy text inputs
    dummy_input_ids = torch.ones((batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
    }
    
    # Add image input if requested for Phi-3 Vision
    if args.with_image_input and hasattr(processor, "image_processor"):
        # Check if processor has image processing capabilities
        try:
            # Create dummy RGB image with expected dimensions
            # Phi-3 Vision uses 448x448 images
            expected_size = 448
            dummy_image = torch.ones(
                (batch_size, 3, expected_size, expected_size), 
                dtype=torch.float32
            )
            inputs["pixel_values"] = dummy_image
        except Exception as e:
            logger.warning(f"Unable to add image inputs: {e}")
    
    return inputs

def export_to_onnx(model, inputs, output_path, args):
    """Export the model to ONNX format"""
    logger.info(f"Exporting model to ONNX format with opset {args.opset_version}")
    
    # Define output names based on model type
    output_names = ["logits"]
    
    # Define dynamic axes
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
    
    # Add dynamic axes for pixel values if present
    if "pixel_values" in inputs:
        dynamic_axes["pixel_values"] = {0: "batch_size"}
    
    # Export the model
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            output_path,
            input_names=list(inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
    
    logger.info(f"Model exported to: {output_path}")
    return output_path

def optimize_onnx_model(onnx_path):
    """Apply ONNX Runtime optimizations to the exported model"""
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
        
        logger.info("Optimizing ONNX model")
        
        # Load the model
        model = onnx.load(onnx_path)
        
        # Configure optimization options
        opt_options = FusionOptions("gpt2")
        opt_options.enable_gelu_approximation = True
        
        # Create optimizer
        opt_model = optimizer.optimize_model(
            onnx_path,
            model_type="gpt2",  # Use GPT2 as the closest model type
            num_heads=model.graph.input[0].type.tensor_type.shape.dim[0].dim_value,
            hidden_size=model.graph.input[0].type.tensor_type.shape.dim[1].dim_value,
            optimization_options=opt_options
        )
        
        # Save optimized model
        opt_model.save_model_to_file(onnx_path)
        
        logger.info("ONNX model optimized successfully")
    except ImportError:
        logger.warning("ONNX optimization failed: onnxruntime.transformers not available")
    except Exception as e:
        logger.warning(f"ONNX optimization failed: {e}")

def quantize_onnx_model(onnx_path):
    """Quantize the ONNX model for reduced size and faster inference"""
    try:
        import onnx
        from onnxruntime.quantization import quantize
        
        logger.info("Quantizing ONNX model")
        
        # Load the model
        model = onnx.load(onnx_path)
        
        # Quantize the model
        quantized_model = quantize(model, per_channel=True, symmetric=True)
        
        # Save quantized model
        onnx.save(quantized_model, onnx_path)
        
        logger.info("ONNX model quantized successfully")
    except ImportError:
        logger.warning("ONNX quantization failed: onnxruntime.quantization not available")
    except Exception as e:
        logger.warning(f"ONNX quantization failed: {e}")

def save_config_files(model_dir, output_dir):
    """Save necessary configuration files for ONNX Runtime"""
    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for file in tokenizer_files:
        src_path = os.path.join(model_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, file))
    
    # Copy model config
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        shutil.copy2(config_path, os.path.join(output_dir, "config.json"))
    
    # Copy processor config if exists
    processor_config_path = os.path.join(model_dir, "processor_config.json")
    if os.path.exists(processor_config_path):
        shutil.copy2(processor_config_path, os.path.join(output_dir, "processor_config.json"))
    
    # Copy any image processor files if they exist
    image_processor_files = ["image_processor_config.json"]
    for file in image_processor_files:
        src_path = os.path.join(model_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, file))
    
    # Create generation config if not exists
    gen_config_path = os.path.join(model_dir, "generation_config.json")
    output_gen_config_path = os.path.join(output_dir, "generation_config.json")
    
    if os.path.exists(gen_config_path):
        shutil.copy2(gen_config_path, output_gen_config_path)
    else:
        # Create a basic generation config
        gen_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
        with open(output_gen_config_path, "w") as f:
            json.dump(gen_config, f, indent=2)

def create_onnx_config(model, output_dir, args):
    """Create ONNX Runtime configuration file"""
    config = {
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "max_seq_length": args.max_seq_length,
        "with_image_input": args.with_image_input,
        "onnx_opset_version": args.opset_version,
    }
    
    config_path = os.path.join(output_dir, "onnx_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ONNX config saved to: {config_path}")

def convert_to_onnx(args):
    """Convert the fine-tuned model to ONNX format"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the fine-tuned model
    logger.info(f"Loading fine-tuned model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    
    # Load model with correct precision
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare dummy inputs for ONNX export
    inputs = prepare_dummy_inputs(model, tokenizer, processor, args)
    
    # Export the model to ONNX
    output_path = os.path.join(args.output_dir, "model.onnx")
    export_to_onnx(model, inputs, output_path, args)
    
    # Optimize ONNX model if requested
    if args.optimize:
        optimize_onnx_model(output_path)
    
    # Quantize ONNX model if requested
    if args.quantize:
        quantize_onnx_model(output_path)
    
    # Save configuration files
    save_config_files(args.model_dir, args.output_dir)
    create_onnx_config(model, args.output_dir, args)
    
    logger.info(f"ONNX conversion complete! Model and configs saved to: {args.output_dir}")
    
    # Create structure folders for ONNX Runtime
    cpu_dir = os.path.join(args.output_dir, "cpu_and_mobile")
    gpu_dir = os.path.join(args.output_dir, "gpu")
    os.makedirs(cpu_dir, exist_ok=True)
    os.makedirs(gpu_dir, exist_ok=True)
    
    # Copy model to appropriate folders
    shutil.copy2(output_path, os.path.join(cpu_dir, "model.onnx"))
    shutil.copy2(output_path, os.path.join(gpu_dir, "model.onnx"))
    
    logger.info("ONNX model prepared for both CPU and GPU deployment")

if __name__ == "__main__":
    args = parse_args()
    convert_to_onnx(args)
