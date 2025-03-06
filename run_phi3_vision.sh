#!/bin/bash
# Phi-3 Vision Fine-Tuning Pipeline
# This script runs the complete pipeline for fine-tuning a Phi-3 Vision model

set -e  # Exit on any error

# Set custom HuggingFace cache directory within our build folder
export HF_HOME="./build/hf_cache"
mkdir -p $HF_HOME

# Set build directory for all artifacts
BUILD_DIR="./build"

# Create build directory structure if it doesn't exist
mkdir -p $BUILD_DIR/{processed_data,fine_tuned_model,onnx_model,evaluation_results/pytorch}
mkdir -p $BUILD_DIR/processed_data/images

# Set directories
DATA_DIR="./sample_data"
PROCESSED_DIR="$BUILD_DIR/processed_data"
OUTPUT_DIR="$BUILD_DIR/fine_tuned_model"
ONNX_DIR="$BUILD_DIR/onnx_model"
EVAL_DIR="$BUILD_DIR/evaluation_results"
echo "===================================================================="
echo "Step 1: Data Preprocessing"
echo "===================================================================="
python 1_preprocess_data.py --data_dir $DATA_DIR --output_dir $PROCESSED_DIR --model_name microsoft/Phi-3-vision-128k-instruct

echo "===================================================================="
echo "Step 2: Fine-tuning the model with LoRA"
echo "===================================================================="
python 2_finetune.py --data_path $PROCESSED_DIR/processed_data.json --output_dir $OUTPUT_DIR

echo "===================================================================="
echo "Step 3: Converting the model to ONNX"
echo "===================================================================="
python 3_convert_to_onnx.py --model_dir $OUTPUT_DIR --output_dir $ONNX_DIR --with_image_input

echo "===================================================================="
echo "Step 4: Evaluating the model"
echo "===================================================================="
python 4_evaluate_model.py --model_dir $OUTPUT_DIR --data_path $DATA_DIR/sample_data.json --output_dir $EVAL_DIR/pytorch

echo "===================================================================="
echo "Step 5: Running inference with the fine-tuned model"
echo "===================================================================="
# Test on the first sample image
python 5_inference.py --model_path $OUTPUT_DIR --image_path $PROCESSED_DIR/images/sample1.jpg --prompt "Describe what you see in this image."

echo "===================================================================="
echo "Pipeline completed successfully!"
echo "All build artifacts are in: $BUILD_DIR"
echo "  - Fine-tuned PyTorch model: $OUTPUT_DIR"
echo "  - ONNX model: $ONNX_DIR"
echo "  - Evaluation results: $EVAL_DIR"
echo "  - Processed data: $PROCESSED_DIR"
echo "===================================================================="