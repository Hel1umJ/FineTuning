"""
Data Generation Script for Phi-3 Vision Fine-tuning

This script creates a sample dataset for testing or demonstrating the fine-tuning pipeline.
It can generate data in two ways:
1. Download sample images from the internet (if internet connection is available)
2. Generate synthetic images locally (if no internet connection or for fully offline testing)

Usage:
    python 0_generate_data.py --mode [download|local] --output_dir ./sample_data --num_samples 5
"""

import os
import json
import argparse
import requests
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create sample data for fine-tuning")
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["download", "local"],
        help="Mode for generating images: 'download' (from internet) or 'local' (synthetic)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_data",
        help="Directory where sample data will be saved"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample entries to create"
    )
    return parser.parse_args()

def download_image(url, save_path):
    """Download an image from a URL and save it to the specified path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        print("Falling back to local image generation...")
        return False

def create_synthetic_image(save_path, image_type="building", size=(448, 448)):
    """Create a simple synthetic image for testing"""
    img = Image.new('RGB', size, color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    draw = ImageDraw.Draw(img)
    
    if image_type == "building":
        # Draw a simple building
        draw.rectangle([(100, 100), (350, 400)], fill=(70, 80, 90), outline=(30, 30, 30))
        # Windows
        for y in range(150, 380, 50):
            for x in range(130, 330, 60):
                draw.rectangle([(x, y), (x+40, y+30)], fill=(173, 216, 230))
        # Door
        draw.rectangle([(200, 320), (250, 400)], fill=(120, 80, 40))
        
    elif image_type == "car":
        # Draw a simple car
        draw.rectangle([(100, 250), (350, 330)], fill=(255, 0, 0), outline=(0, 0, 0))  # Body
        draw.rectangle([(130, 180), (320, 250)], fill=(255, 0, 0), outline=(0, 0, 0))  # Top
        # Wheels
        draw.ellipse([(130, 310), (180, 360)], fill=(30, 30, 30))
        draw.ellipse([(270, 310), (320, 360)], fill=(30, 30, 30))
        # Windows
        draw.rectangle([(140, 190), (200, 240)], fill=(173, 216, 230))
        draw.rectangle([(210, 190), (310, 240)], fill=(173, 216, 230))
        
    elif image_type == "construction":
        # Draw construction site
        draw.rectangle([(100, 300), (350, 400)], fill=(139, 69, 19))  # Ground
        # Draw a crane
        draw.line([(200, 100), (200, 350)], fill=(255, 140, 0), width=10)  # Vertical
        draw.line([(200, 100), (300, 100)], fill=(255, 140, 0), width=10)  # Horizontal
        draw.line([(300, 100), (300, 150)], fill=(0, 0, 0), width=2)  # Cable
        draw.rectangle([(280, 150), (320, 190)], fill=(128, 128, 128))  # Load
        # Safety cones
        for x in range(120, 320, 50):
            draw.polygon([(x, 350), (x+20, 350), (x+10, 320)], fill=(255, 140, 0))
            
    elif image_type == "machine":
        # Draw industrial equipment
        draw.rectangle([(150, 150), (300, 350)], fill=(192, 192, 192), outline=(100, 100, 100), width=3)
        # Control panel
        draw.rectangle([(170, 180), (280, 230)], fill=(50, 50, 50))
        # Buttons and dials
        for x in range(185, 270, 30):
            draw.ellipse([(x, 190), (x+20, 210)], fill=random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)]))
        # Pipes
        draw.rectangle([(200, 350), (220, 400)], fill=(100, 100, 100))
        draw.rectangle([(230, 350), (250, 400)], fill=(100, 100, 100))
        # Rust spots
        for _ in range(10):
            x, y = random.randint(150, 300), random.randint(250, 350)
            draw.ellipse([(x, y), (x+15, y+10)], fill=(139, 69, 19))
            
    elif image_type == "scene":
        # Draw a simple landscape
        # Sky
        draw.rectangle([(0, 0), (448, 250)], fill=(135, 206, 235))
        # Ground
        draw.rectangle([(0, 250), (448, 448)], fill=(34, 139, 34))
        # Sun
        draw.ellipse([(350, 50), (420, 120)], fill=(255, 255, 0))
        # Trees
        for x in range(50, 400, 120):
            draw.rectangle([(x-5, 280), (x+5, 330)], fill=(139, 69, 19))
            draw.ellipse([(x-30, 200), (x+30, 280)], fill=(0, 100, 0))
    
    # Add some text
    text = f"Sample {image_type.title()}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((size[0]//2-50, 30), text, fill=(0, 0, 0), font=font)
    
    # Add some noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Save the image
    img.save(save_path)
    return True

def generate_data(args):
    """Create a sample dataset with images and prompts"""
    # Create output directory and subdirectories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    
    output_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Sample image URLs (public domain or CC0 license images)
    sample_images = [
        "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg",  # Office building
        "https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg",    # Car
        "https://images.pexels.com/photos/159306/construction-site-build-construction-work-159306.jpeg",  # Construction site
        "https://images.pexels.com/photos/162568/oil-pump-jack-sunset-clouds-silhouette-162568.jpeg",     # Industrial equipment
        "https://images.pexels.com/photos/355948/pexels-photo-355948.jpeg",    # General scene
    ]
    
    # Image types for local generation
    image_types = ["building", "car", "construction", "machine", "scene"]
    
    # Sample prompts for Phi-3 Vision
    sample_prompts = [
        "Describe what you see in this image. <image>",
        "What is unusual about this image? <image>",
        "Create a safety report based on this image. <image>",
        "What maintenance issues can you identify in this image? <image>",
        "Describe this landscape scene. <image>",
    ]
    
    # Sample completions
    sample_completions = [
        "I can see a modern office building with glass facades reflecting the blue sky. The architecture features clean lines and a minimalist design approach typical of contemporary corporate structures.",
        "The unusual aspect of this image is that the car appears to be floating above the road. This is likely a digital manipulation or optical illusion created by the angle of the photograph.",
        "Safety Report: The image shows a construction site with several safety concerns. Workers are not wearing proper PPE (personal protective equipment), specifically missing hard hats and high-visibility vests. There's also unsecured scaffolding and trip hazards visible on the ground. Immediate remediation is recommended including: 1) Ensuring all workers wear complete PPE, 2) Securing all scaffolding properly, 3) Clearing walkways of debris and trip hazards.",
        "The maintenance issues visible in this image include: 1) Significant corrosion on the metal components, 2) A loose connection at the joint indicated by the visible gap, 3) Missing fasteners on the left side bracket, 4) Signs of fluid leakage visible at the bottom of the component, and 5) Improper alignment of the mounting assembly. These issues require immediate attention to prevent equipment failure and ensure operational safety.",
        "This image shows a peaceful landscape scene with a bright blue sky occupying the upper portion. In the foreground is a lush green field. The scene includes several evenly spaced trees with brown trunks and rounded green foliage. A bright yellow sun is visible in the upper right corner of the sky, creating a cheerful daytime atmosphere. The overall composition conveys a tranquil natural setting.",
    ]
    
    # Text-only example
    text_prompt = "Is there a text-only question you can answer?"
    text_completion = "Yes, I can answer text-only questions without an image input. For example, I can provide information about various topics, help with problem-solving, or offer explanations on different subjects based solely on the text prompt provided."
    
    # Create sample data entries
    data = []
    print(f"Generating {args.num_samples} sample data entries...")
    
    # Generate image examples
    for i in range(min(args.num_samples, len(sample_prompts))):
        image_path = images_dir / f"sample{i+1}.jpg"
        success = False
        
        # Try to download image if in download mode
        if args.mode == "download":
            if i < len(sample_images):
                print(f"Downloading image {i+1}...")
                success = download_image(sample_images[i], image_path)
        
        # If download failed or in local mode, generate synthetic image
        if not success:
            print(f"Generating synthetic image {i+1}...")
            create_synthetic_image(image_path, image_type=image_types[i % len(image_types)])
            success = True
        
        # Create data entry
        if success:
            entry = {
                "prompt": sample_prompts[i],
                "image_path": f"images/sample{i+1}.jpg",
                "completion": sample_completions[i]
            }
            data.append(entry)
    
    # Add text-only example
    data.append({
        "prompt": text_prompt,
        "image_path": None,
        "completion": text_completion
    })
    
    # Save the data to a JSON file
    data_path = output_dir / "sample_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Created sample dataset with {len(data)} entries")
    print(f"Data saved to: {data_path}")
    print(f"Images saved to: {images_dir}")

if __name__ == "__main__":
    args = parse_args()
    generate_data(args)
