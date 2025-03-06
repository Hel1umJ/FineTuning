"""
Model Evaluation Script for Phi-3 Vision

This script evaluates a fine-tuned Phi-3 Vision model on a test dataset.
It computes various metrics including BLEU, ROUGE, and embedding similarity
to assess model performance. It can evaluate both PyTorch and ONNX models.

Usage:
    # Evaluate PyTorch model
    python 4_evaluate_model.py --model_dir ./build/fine_tuned_model --data_path ./sample_data/sample_data.json --output_dir ./build/evaluation_results
    
    # Evaluate ONNX model
    python 4_evaluate_model.py --onnx_model_dir ./build/onnx_model --data_path ./sample_data/sample_data.json --output_dir ./build/evaluation_results/onnx
"""

import os
import json
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phi-3 Vision model performance")
    
    # Model inputs (PyTorch or ONNX)
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument(
        "--model_dir",
        type=str,
        default="./build/fine_tuned_model",
        help="Directory containing the fine-tuned PyTorch model"
    )
    model_group.add_argument(
        "--onnx_model_dir",
        type=str,
        default="./build/onnx_model",
        help="Directory containing the ONNX model"
    )
    
    # Data and output settings
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the test data JSON file"
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing the fine-tuned PyTorch model"
    )
    model_group.add_argument(
        "--onnx_model_dir",
        type=str,
        help="Directory containing the fine-tuned ONNX model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the test data JSON file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Base directory for image paths (if different from data_path directory)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    
    return parser.parse_args()

def load_data(data_path, data_dir=None):
    """Load test data from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process image paths if data_dir is provided
    if data_dir is not None:
        for item in data:
            if item.get("image_path"):
                item["image_path"] = os.path.join(data_dir, item["image_path"])
    else:
        # Use the directory of the data_path as the base for relative image paths
        base_dir = os.path.dirname(data_path)
        for item in data:
            if item.get("image_path"):
                item["image_path"] = os.path.join(base_dir, item["image_path"])
    
    return data

def init_nltk():
    """Initialize NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def load_pytorch_model(model_dir):
    """Load fine-tuned Phi model from directory"""
    print(f"Loading PyTorch model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer, processor

def load_onnx_model(onnx_model_dir):
    """Load ONNX model for generation"""
    if not ONNXRUNTIME_GENAI_AVAILABLE:
        raise ImportError("onnxruntime_genai is required for ONNX model evaluation")
    
    print(f"Loading ONNX model from {onnx_model_dir}...")
    
    # Determine if CPU or GPU should be used
    providers = None
    execution_provider = None
    if torch.cuda.is_available():
        # Check for GPU directory
        gpu_dir = os.path.join(onnx_model_dir, "gpu")
        if os.path.exists(gpu_dir) and os.path.isdir(gpu_dir):
            onnx_model_dir = gpu_dir
            execution_provider = "gpu"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            # Fall back to CPU
            cpu_dir = os.path.join(onnx_model_dir, "cpu_and_mobile")
            if os.path.exists(cpu_dir) and os.path.isdir(cpu_dir):
                onnx_model_dir = cpu_dir
            execution_provider = "cpu"
    else:
        # Use CPU
        cpu_dir = os.path.join(onnx_model_dir, "cpu_and_mobile")
        if os.path.exists(cpu_dir) and os.path.isdir(cpu_dir):
            onnx_model_dir = cpu_dir
        execution_provider = "cpu"
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(onnx_model_dir)
    processor = None
    processor_config_path = os.path.join(onnx_model_dir, "processor_config.json")
    if os.path.exists(processor_config_path):
        processor = AutoProcessor.from_pretrained(onnx_model_dir)
    
    # Create model
    model = og.Model.from_pretrained(
        model_id=onnx_model_dir,
        providers=providers,
    )
    
    return model, tokenizer, processor, execution_provider

def generate_with_pytorch(model, tokenizer, processor, prompt, image_path=None, max_new_tokens=128):
    """Generate text using PyTorch model"""
    inputs = {}
    
    # Process image if provided
    if image_path and processor:
        image = Image.open(image_path).convert('RGB')
        image_inputs = processor(images=image, return_tensors="pt")
        
        # Move pixel values to the same device as model
        pixel_values = image_inputs.pixel_values.to(model.device)
        inputs["pixel_values"] = pixel_values
    
    # Process text
    encoded_text = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_text.input_ids.to(model.device)
    attention_mask = encoded_text.attention_mask.to(model.device)
    
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the completion part (after the prompt)
    completion = generated_text[len(prompt):].strip()
    return completion

def generate_with_onnx(model, tokenizer, processor, prompt, image_path=None, max_new_tokens=128):
    """Generate text using ONNX model"""
    # Prepare request
    request = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    # Add image if provided
    if image_path and processor:
        try:
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            # Process the image using the processor
            request["images"] = [image]
        except Exception as e:
            print(f"Error processing image: {e}")
    
    # Generate text
    result = model.generate(**request)
    generated_text = result.text
    
    # Extract the completion part (after the prompt)
    completion = generated_text[len(prompt):].strip()
    return completion

def compute_metrics(reference, hypothesis, sentence_model=None):
    """Compute evaluation metrics between reference and hypothesis"""
    metrics = {}
    
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # BLEU
    smoothing = SmoothingFunction().method1
    try:
        bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        metrics["bleu1"] = bleu1
        metrics["bleu4"] = bleu4
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        metrics["bleu1"] = 0.0
        metrics["bleu4"] = 0.0
    
    # ROUGE
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, hypothesis)
        metrics["rouge1"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        metrics["rouge1"] = 0.0
        metrics["rouge2"] = 0.0
        metrics["rougeL"] = 0.0
    
    # Embedding similarity if available
    if sentence_model is not None:
        try:
            # Compute embeddings
            ref_embedding = sentence_model.encode([reference])
            hyp_embedding = sentence_model.encode([hypothesis])
            
            # Compute cosine similarity
            similarity = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
            metrics["embedding_similarity"] = similarity
        except Exception as e:
            print(f"Error computing embedding similarity: {e}")
            metrics["embedding_similarity"] = 0.0
    
    return metrics

def evaluate_model(args):
    """Evaluate the model on the test data"""
    # Initialize NLTK
    init_nltk()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading test data from {args.data_path}...")
    test_data = load_data(args.data_path, args.data_dir)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize sentence transformer for embedding comparison
    sentence_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded sentence-transformers model for embedding similarity")
        except Exception as e:
            print(f"Error loading sentence-transformers model: {e}")
    
    # Load model (PyTorch or ONNX)
    is_onnx = args.onnx_model_dir is not None
    if is_onnx:
        if not ONNXRUNTIME_GENAI_AVAILABLE:
            raise ImportError("onnxruntime_genai is required for ONNX model evaluation")
        model, tokenizer, processor, execution_provider = load_onnx_model(args.onnx_model_dir)
        print(f"Loaded ONNX model with {execution_provider} execution provider")
        model_type = "onnx"
        model_dir = args.onnx_model_dir
    else:
        model, tokenizer, processor = load_pytorch_model(args.model_dir)
        print(f"Loaded PyTorch model on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        model_type = "pytorch"
        model_dir = args.model_dir
    
    # Evaluate model
    results = []
    all_metrics = []
    
    print(f"Evaluating model on {len(test_data)} examples...")
    for idx, example in enumerate(tqdm(test_data)):
        # Extract data from example
        prompt = example["prompt"]
        image_path = example.get("image_path")
        reference = example["completion"]
        
        # Prepare actual image path if needed
        if image_path is not None:
            if not os.path.exists(image_path) and not os.path.isabs(image_path):
                # Try relative to data_dir or model_dir
                for base_dir in [args.data_dir, os.path.dirname(args.data_path), model_dir]:
                    if base_dir is not None:
                        potential_path = os.path.join(base_dir, image_path)
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
        
        # Generate completion
        try:
            if is_onnx:
                hypothesis = generate_with_onnx(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt=prompt,
                    image_path=image_path,
                    max_new_tokens=args.max_new_tokens
                )
            else:
                hypothesis = generate_with_pytorch(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt=prompt,
                    image_path=image_path,
                    max_new_tokens=args.max_new_tokens
                )
        except Exception as e:
            print(f"Error generating completion for example {idx}: {e}")
            hypothesis = ""
        
        # Compute metrics
        metrics = compute_metrics(reference, hypothesis, sentence_model)
        all_metrics.append(metrics)
        
        # Save result
        results.append({
            "example_id": idx,
            "prompt": prompt,
            "image_path": image_path,
            "reference": reference,
            "hypothesis": hypothesis,
            "metrics": metrics
        })
    
    # Compute average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics if m[metric] is not None]
        if values:
            avg_metrics[metric] = sum(values) / len(values)
        else:
            avg_metrics[metric] = None
    
    # Save results
    output_path = os.path.join(args.output_dir, f"evaluation_results_{model_type}.json")
    output_data = {
        "model_type": model_type,
        "model_path": model_dir,
        "data_path": args.data_path,
        "num_examples": len(test_data),
        "average_metrics": avg_metrics,
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Evaluation complete! Results saved to: {output_path}")
    print("Average metrics:")
    for metric, value in avg_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
