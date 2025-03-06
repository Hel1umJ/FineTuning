# Phi-3 Vision Fine-tuning Pipeline Script
# This script runs the entire pipeline from data preparation to evaluation

# Configuration parameters
$DATA_DIR = ".\sample_data"
$PROCESSED_DIR = ".\processed_data"
$FINETUNED_DIR = ".\fine_tuned_model"
$ONNX_DIR = ".\onnx_model"
$EVAL_DIR = ".\evaluation_results"
$MODEL_NAME = "microsoft/phi-3-vision-instruct"
$SAMPLE_DATA = "sample_data_phi3.json"

# Training parameters (can be modified for your specific needs)
$MAX_STEPS = 50
$BATCH_SIZE = 4
$LEARNING_RATE = 5e-5
$LOGGING_STEPS = 5
$SAVE_STEPS = 10

# Create a local cache directory with proper permissions
if (-not (Test-Path ".\hf_cache")) {
    New-Item -ItemType Directory -Path ".\hf_cache"
}
$env:HF_HOME = ".\hf_cache"
$env:TRANSFORMERS_CACHE = ".\hf_cache"

# Step 0: Ensure directories exist
foreach ($dir in @($PROCESSED_DIR, $FINETUNED_DIR, $ONNX_DIR, $EVAL_DIR)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir
    }
}

# Step 1: Install dependencies
Write-Host "Step 1: Installing dependencies..." -ForegroundColor Green

# First install torch separately to ensure it's available
pip install "torch>=2.0.0" --quiet

# Then install other dependencies
$dependencies = @(
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "pillow",
    "tqdm",
    "onnx",
    "onnxruntime",
    "onnxruntime-genai",
    "sentencepiece",
    "nltk",
    "rouge_score",
    "sentence-transformers",
    "torchvision",
    "backoff"
)

foreach ($dep in $dependencies) {
    Write-Host "Installing $dep..." -ForegroundColor Yellow
    pip install $dep --quiet
}

# Try to install flash-attn if possible (may fail on some systems)
Write-Host "Attempting to install flash-attn..." -ForegroundColor Yellow
try {
    pip install flash-attn --no-build-isolation --quiet
} catch {
    Write-Host "Could not install flash-attn, continuing without it" -ForegroundColor Yellow
}

# Step 2: Create sample data if needed
if (-not (Test-Path "$DATA_DIR\$SAMPLE_DATA")) {
    Write-Host "Step 2: Creating sample data..." -ForegroundColor Green
    python "0_generate_data.py" --mode local --output_dir $DATA_DIR
} else {
    Write-Host "Step 2: Using existing sample data..." -ForegroundColor Green
}

# Step 3: Preprocess data
Write-Host "Step 3: Running data preprocessing..." -ForegroundColor Green
python "1_preprocess_data.py" --data_dir $DATA_DIR --output_dir $PROCESSED_DIR --model_name $MODEL_NAME

# Step 4: Fine-tune model with LoRA
Write-Host "Step 4: Fine-tuning the model with LoRA..." -ForegroundColor Green
python "2_finetune.py" `
    --data_dir $PROCESSED_DIR `
    --output_dir $FINETUNED_DIR `
    --model_name $MODEL_NAME `
    --use_lora `
    --max_steps $MAX_STEPS `
    --per_device_train_batch_size $BATCH_SIZE `
    --logging_steps $LOGGING_STEPS `
    --save_steps $SAVE_STEPS `
    --learning_rate $LEARNING_RATE

# Step 5: Convert to ONNX format
Write-Host "Step 5: Converting to ONNX format..." -ForegroundColor Green
python "3_convert_to_onnx.py" --model_dir $FINETUNED_DIR --output_dir $ONNX_DIR --with_image_input --optimize

# Step 6: Evaluate the fine-tuned PyTorch model
Write-Host "Step 6: Evaluating the fine-tuned PyTorch model..." -ForegroundColor Green
python "4_evaluate_model.py" --model_dir $FINETUNED_DIR --data_path "$DATA_DIR\$SAMPLE_DATA" --output_dir "$EVAL_DIR\pytorch_evaluation"

# Step 7: Evaluate the ONNX model
Write-Host "Step 7: Evaluating the ONNX model..." -ForegroundColor Green
python "4_evaluate_model.py" --onnx_model_dir $ONNX_DIR --data_path "$DATA_DIR\$SAMPLE_DATA" --output_dir "$EVAL_DIR\onnx_evaluation"

# Done
Write-Host "Complete! The Phi-3 Vision fine-tuning pipeline has been executed." -ForegroundColor Green
Write-Host "Locations:" -ForegroundColor Yellow
Write-Host "  - Processed data: $PROCESSED_DIR" -ForegroundColor Yellow
Write-Host "  - Fine-tuned model: $FINETUNED_DIR" -ForegroundColor Yellow
Write-Host "  - ONNX model: $ONNX_DIR" -ForegroundColor Yellow
Write-Host "Evaluation results:" -ForegroundColor Yellow
Write-Host "  - PyTorch: $EVAL_DIR\pytorch_evaluation\evaluation_results_pytorch.json" -ForegroundColor Yellow
Write-Host "  - ONNX: $EVAL_DIR\onnx_evaluation\evaluation_results_onnx.json" -ForegroundColor Yellow
