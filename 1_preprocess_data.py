"""
Data Preprocessing Script for Phi-3 Vision Fine-tuning

This script processes a dataset of text and images into the format expected by the 
Phi-3 Vision fine-tuning pipeline. It handles:
1. Image resizing and normalization
2. Text tokenization
3. Creating a JSON dataset in the expected format

Usage:
    python 1_preprocess_data.py --data_dir ./your_data --output_dir ./processed_data
"""
import os
import json
import argparse
import shutil
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for Phi-3 Vision fine-tuning")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing your raw data (images and text)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./build/processed_data",
        help="Directory where processed data will be saved"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-vision-128k-instruct",
        help="Model name to use for tokenization/processing"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,  # This is default for Phi-3-vision
        help="Size to resize images to (square)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization"
    )
    return parser.parse_args()

class DataFormatter:
    """Handles formatting the data for fine-tuning"""
    
    def __init__(self, model_name: str, image_size: int, max_length: int):
        self.model_name = model_name
        self.image_size = image_size
        self.max_length = max_length
        
        logger.info(f"Loading processor and tokenizer from {model_name}...")
        try:
            cache_dir = os.environ.get("HF_HOME", None)
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            # Set padding_side to 'left' for Flash Attention compatibility
            self.tokenizer.padding_side = 'left'
            logger.info("Tokenizer initialized with padding_side='left' for Flash Attention compatibility")
            logger.info("Successfully loaded processor and tokenizer")
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise

    def process_example(self, text: str, image_path: Optional[str] = None):
        """Process a single example with text and optional image"""
        result = {
            "text": text,
            "image_path": image_path
        }
        
        # Ensure image path exists if provided
        if image_path and not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {image_path}")
            result["image_path"] = None
        
        return result
        
    def tokenize_text(self, text: str):
        """Tokenize text input"""
        # Ensure padding_side is 'left' for Flash Attention compatibility
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = 'left'
        return self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length)

def read_data_format(data_dir: str) -> List[Dict[str, Any]]:
    """
    Reads data from a directory. Expects one of these formats:
    1. A JSON file with entries containing "prompt" and optional "image_path"
    2. A directory with subdirectory for images, and a text file with prompts
    
    Returns a list of dictionaries with {"text": str, "image_path": str or None}
    """
    data_path = Path(data_dir)
    examples = []
    
    # Check if there's a JSON file with data
    json_files = list(data_path.glob("*.json"))
    if json_files:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            if isinstance(item, dict) and "prompt" in item:
                image_path = None
                if "image_path" in item and item["image_path"]:
                    image_path = os.path.join(data_dir, item["image_path"])
                examples.append({
                    "text": item["prompt"],
                    "image_path": image_path
                })
    
    # If no JSON file found, try to find text files and image directories
    else:
        txt_files = list(data_path.glob("*.txt"))
        if txt_files:
            with open(txt_files[0], "r", encoding="utf-8") as f:
                texts = f.read().strip().split("\n\n")
            
            # Assume images are in 'images' subdirectory
            img_dir = data_path / "images"
            if img_dir.exists():
                img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                
                # If number of text entries matches images, pair them
                if len(texts) == len(img_files):
                    for text, img in zip(texts, img_files):
                        examples.append({
                            "text": text,
                            "image_path": str(img)
                        })
                else:
                    # Otherwise, just use the texts without images
                    for text in texts:
                        examples.append({
                            "text": text,
                            "image_path": None
                        })
            else:
                # No images found, just use texts
                for text in texts:
                    examples.append({
                        "text": text,
                        "image_path": None
                    })
    
    return examples

def preprocess_data(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create images directory in output
    img_output_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Initialize formatter
    formatter = DataFormatter(
        model_name=args.model_name,
        image_size=args.image_size,
        max_length=args.max_length
    )
    
    # Read and process the data
    examples = read_data_format(args.data_dir)
    processed_examples = []
    
    print(f"Processing {len(examples)} examples...")
    for example in tqdm(examples):
        processed = formatter.process_example(
            text=example["text"],
            image_path=example["image_path"]
        )
        
        # Copy image to output directory if it exists
        if processed["image_path"]:
            img_filename = os.path.basename(processed["image_path"])
            img_dest_path = os.path.join(img_output_dir, img_filename)
            shutil.copy2(processed["image_path"], img_dest_path)
            processed["image_path"] = os.path.join("images", img_filename)
        
        processed_examples.append(processed)
    
    # Save the processed examples to a JSON file
    output_file = os.path.join(args.output_dir, "processed_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_examples, f, indent=2)
    
    print(f"Preprocessing complete! Processed {len(processed_examples)} examples.")
    print(f"Output saved to: {output_file}")
    print(f"Images saved to: {img_output_dir}")

if __name__ == "__main__":
    args = parse_args()
    preprocess_data(args)
