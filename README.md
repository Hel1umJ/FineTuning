# Phi-3 Vision Fine-Tuning Pipeline

This project provides a complete pipeline for fine-tuning the Microsoft Phi-3 Vision model on custom datasets. The pipeline is designed to be easy to use and adaptable to different use cases.

## Overview

The Phi-3 Vision model is a lightweight, state-of-the-art multimodal model that can process both text and images. This pipeline allows you to fine-tune the model on your own dataset, making it more effective for specific tasks like image description, defect identification, or visual question answering.

## Pipeline Structure

The project is organized as a step-by-step pipeline:

1. **Data Preprocessing** (`1_preprocess_data.py`): Prepares your raw data for fine-tuning
2. **Fine-tuning** (`2_finetune.py`): Fine-tunes the model using LoRA for efficiency
3. **ONNX Conversion** (`3_convert_to_onnx.py`): Converts the fine-tuned model to ONNX format
4. **Model Evaluation** (`4_evaluate_model.py`): Evaluates the model performance
5. **Inference** (`5_inference.py`): Tests the fine-tuned model on new images

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- flash-attn (for GPU acceleration)

## Quick Start

To run the complete pipeline:

```bash
./run_phi3_vision.sh
```

Or run each step individually:

```bash
# Step 1: Preprocess data
python 1_preprocess_data.py --data_dir ./sample_data --output_dir ./build/processed_data

# Step 2: Fine-tune model
python 2_finetune.py --data_path ./build/processed_data/processed_data.json --output_dir ./build/fine_tuned_model

# Step 3: Convert to ONNX
python 3_convert_to_onnx.py --model_dir ./build/fine_tuned_model --output_dir ./build/onnx_model --with_image_input

# Step 4: Evaluate model
python 4_evaluate_model.py --model_dir ./build/fine_tuned_model --data_path ./sample_data/sample_data.json --output_dir ./build/evaluation_results

# Step 5: Run inference
python 5_inference.py --model_path ./build/fine_tuned_model --image_path ./build/processed_data/images/sample1.jpg
```

All build artifacts (processed data, fine-tuned models, ONNX models, and evaluation results) are stored in the `build` directory to keep the project root clean.

## Data Format

The pipeline expects input data in a specific format:

### Input Format (for Step 1)

The pipeline expects a file named `sample_data.json` in the `sample_data` directory with the following format:

```json
[
  {
    "prompt": "Describe what you see in this image. <image_1>",
    "image_path": "images/image1.jpg",
    "completion": "This is a modern office building with glass facades..."
  },
  {
    "prompt": "What is unusual about this image? <image_1>",
    "image_path": "images/image2.jpg",
    "completion": "The unusual aspect is that the car appears to be floating..."
  }
]
```

Notes:
- The `prompt` field contains the text prompt to the model
- The `image_path` field is relative to the sample_data directory
- The `completion` field contains the expected response for fine-tuning

### Processed Format (after Step 1)

The preprocessed data follows this format:

```json
[
  {
    "text": "Describe what you see in this image. <image_1>",
    "image_path": "images/image1.jpg"
  }
]
```

## Fine-tuning Parameters

Key parameters for fine-tuning (Step 2):

- `--data_path`: Path to the preprocessed data
- `--output_dir`: Directory to save the fine-tuned model
- `--lora_r` and `--lora_alpha`: LoRA parameters (default: r=16, alpha=32)
- `--max_steps`: Number of training steps (default: 100)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--per_device_train_batch_size`: Batch size per GPU (default: 1)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 4)

## Inference

Run inference on new images with:

```bash
python 5_inference.py \
    --model_path ./build/fine_tuned_model \
    --image_path path/to/your/image.jpg \
    --prompt "Describe what you see in this image."
```

## ONNX Model and Evaluation

The pipeline also includes:

- Converting to ONNX format for deployment (Step 3)
- Evaluating model performance (Step 4)

These steps prepare your model for production use while providing metrics on its performance.

## Implementation Details

- **LoRA Fine-tuning**: Uses Parameter-Efficient Fine-Tuning to reduce memory requirements
- **Flash Attention**: Automatically enables Flash Attention 2 when running on compatible CUDA GPUs
- **Data Handling**: Robust data processing that handles both text and images correctly
- **Chat Format**: Uses proper chat templates for the Phi-3 Vision model

## Troubleshooting

- **Memory Issues**: Reduce batch size or use gradient accumulation
- **CUDA Problems**: Make sure PyTorch is installed with CUDA support
- **Flash Attention**: Install flash-attn properly for your CUDA version
- **Permission Issues**: The pipeline automatically sets up a custom HuggingFace cache directory in the build folder to avoid permission problems
- **Padding Direction**: All tokenizers are configured with `padding_side='left'` for compatibility with Flash Attention in Phi-3 models

## License

This project uses the Microsoft Phi-3 Vision model, which has its own license. Please refer to the [Microsoft Phi-3 license](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) for details.