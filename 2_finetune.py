"""
Fine-tuning Script for Phi-3 Vision (Step 2)

This script fine-tunes the Phi-3 Vision model on a custom dataset prepared by the preprocessing script.
It supports LoRA fine-tuning for efficient training with limited GPU resources.

Usage:
    python 2_finetune.py --data_path ./processed_data/processed_data.json --output_dir ./fine_tuned_model
"""

import os
import json
import logging
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from PIL import Image
import numpy as np

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 Vision model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./build/processed_data/processed_data.json",
        help="Path to the preprocessed data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./build/fine_tuned_model",
        help="Directory where the fine-tuned model will be saved"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-vision-128k-instruct",
        help="Phi-3 Vision model name or path"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before backward pass"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps during training"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoints every N steps"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def load_data_from_json(data_path, base_path=""):
    """Load data from JSON file and convert to format expected by the model"""
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Prepare for conversion to DataFrame
    processed_data = []
    
    for example in data:
        image_path = example.get("image_path")
        if image_path:
            # Handle the image path - join with base path if needed
            if base_path:
                # Handle path joining based on the provided image_path structure
                image_path = os.path.join(os.path.dirname(data_path), image_path)
            
            # Verify image exists
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
        
        # Create prompt and assistant response format
        # For Phi-3 Vision, we use the <image> tag to indicate image placement
        prompt = example["text"]
        if "<image>" not in prompt and image_path:
            prompt = "<image>\n" + prompt
        
        # Add a simple assistant response for fine-tuning
        # In a real scenario, you would have ground truth responses
        assistant_response = "This is a placeholder response for fine-tuning."
        
        processed_data.append({
            "userPrompt": prompt,
            "imageURL": image_path,
            "assistantResponse": assistant_response
        })
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(processed_data)
    return df

def preprocess_for_phi3_vision(examples, processor, tokenizer, max_seq_length=512):
    """Preprocess examples for Phi-3 Vision"""
    chat_template = """<|user|>
{prompt}
<|assistant|>
{response}"""
    
    images = []
    texts = []
    
    for i in range(len(examples["userPrompt"])):
        prompt = examples["userPrompt"][i]
        response = examples["assistantResponse"][i]
        image_path = examples["imageURL"][i]
        
        # Load image if available
        image = None
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
        
        # Format text using chat template
        formatted_text = chat_template.format(prompt=prompt, response=response)
        
        images.append(image)
        texts.append(formatted_text)
    
    # Process images
    if any(img is not None for img in images):
        # Replace None with an empty black image to keep batch processing consistent
        for i, img in enumerate(images):
            if img is None:
                # Create a black 1x1 pixel image
                images[i] = Image.new('RGB', (448, 448), color='black')
        
        # Process images using the processor
        image_features = processor(images=images, return_tensors="pt")
        pixel_values = image_features.pixel_values
    else:
        pixel_values = None
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Prepare model inputs and labels
    model_inputs = {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
    }
    
    # Add pixel values if available
    if pixel_values is not None:
        model_inputs["pixel_values"] = pixel_values
    
    # Create labels (for causal LM, labels are the same as input_ids)
    # But we mask out the prompt part for loss calculation
    labels = encodings.input_ids.clone()
    
    # Find the positions of <|assistant|> in each sequence
    assistant_token_id = tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]
    assistant_positions = (labels == assistant_token_id).nonzero(as_tuple=True)[1]
    
    # Mask positions before the assistant token (i.e., the user prompt)
    for i, pos in enumerate(assistant_positions):
        labels[i, :pos] = -100  # -100 is the ignore index in CrossEntropyLoss
    
    model_inputs["labels"] = labels
    
    return model_inputs

def fine_tune_model(args):
    """Main function to fine-tune the model"""
    logger.info("Starting fine-tuning process...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Load and prepare the dataset
    logger.info(f"Loading data from {args.data_path}...")
    raw_data = load_data_from_json(args.data_path)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(raw_data)
    
    # Load model components
    logger.info(f"Loading model from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Configure model loading parameters
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    
    # Setup for CUDA if available
    if torch.cuda.is_available():
        model_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
        })
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    
    # Prepare the model for LoRA fine-tuning
    logger.info("Setting up LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA config
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Apply LoRA config to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Preprocess the dataset
    logger.info("Preprocessing dataset...")
    
    # Define preprocessing function using closure to capture the processor and tokenizer
    def preprocess_function(examples):
        return preprocess_for_phi3_vision(
            examples,
            processor=processor,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )
    
    # Process the dataset
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=4,  # Process 4 examples at a time
        remove_columns=dataset.column_names  # Remove original columns after processing
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,  # Important for custom datasets
        seed=args.seed,
        data_seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        dataloader_pin_memory=False,  # Avoid CUDA tensor pinning issues
        group_by_length=False,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model, tokenizer, and processor
    logger.info(f"Saving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save the LoRA adapter separately
    logger.info("Saving LoRA adapter...")
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    args = parse_args()
    fine_tune_model(args)