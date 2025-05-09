import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import gc
import time
import sys
import warnings


def match_images(folder1, folder2, top_n=5):
    """
    Match images between two folders using ImageBind embeddings.

    Args:
        folder1: Path to first folder containing images
        folder2: Path to second folder containing images
        top_n: Number of top matches to return

    Returns:
        Dictionary containing matches, processing time, and memory usage
    """
    start_memory = get_memory_usage()

    # Use MPS (Metal Performance Shaders) if available on Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Try to import ImageBind - if it fails, use a fallback
    try:
        # Install ImageBind dependencies if needed
        import_or_install('git+https://github.com/facebookresearch/ImageBind.git')
        from imagebind import data, models

        # Load ImageBind model
        model = models.imagebind_huge(pretrained=True)
        model = model.to(device)
        model.eval()

        # Define image transformation for ImageBind
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        # Function to extract features using ImageBind
        def extract_features(image_path):
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                # Create inputs for ImageBind
                inputs = {
                    'image': image_tensor
                }

                with torch.no_grad():
                    embeddings = model(inputs)

                return embeddings['image'].cpu().numpy()

            except Exception as e:
                print(f"Error in extract_features for {image_path}: {e}")
                return None

    except Exception as e:
        print(f"Could not load ImageBind: {e}")
        print("Using a fallback model: CLIP...")

        # Fallback to CLIP if ImageBind fails
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Function to extract features using CLIP as fallback
        def extract_features(image_path):
            try:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image)
                return features.cpu().numpy()
            except Exception as e:
                print(f"Error in extract_features for {image_path}: {e}")
                return None

    # Get image files
    img_files1 = get_image_files(folder1)
    img_files2 = get_image_files(folder2)

    print(f"Processing {len(img_files1)} images from folder 1")
    print(f"Processing {len(img_files2)} images from folder 2")

    # Extract features for folder 1
    features1 = {}
    for img_path in tqdm(img_files1, desc="Extracting features from folder 1"):
        try:
            feat = extract_features(img_path)
            if feat is not None:
                features1[img_path] = feat
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Extract features for folder 2
    features2 = {}
    for img_path in tqdm(img_files2, desc="Extracting features from folder 2"):
        try:
            feat = extract_features(img_path)
            if feat is not None:
                features2[img_path] = feat
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Compute similarities between all pairs
    similarities = []
    for path1, feat1 in tqdm(features1.items(), desc="Computing similarities"):
        for path2, feat2 in features2.items():
            try:
                sim = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0][0]
                similarities.append((path1, path2, sim))
            except Exception as e:
                print(f"Error computing similarity between {path1} and {path2}: {e}")

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Take top N matches
    top_matches = similarities[:top_n]

    # Clean up to free memory
    del model
    del features1
    del features2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory

    return {
        "matches": top_matches,
        "memory_usage": memory_used
    }


def import_or_install(package):
    """Import a package, or install it if it's not available."""
    try:
        __import__(package)
    except ImportError:
        import subprocess
        import sys

        # Try to install with pip
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def get_image_files(folder):
    """Get all image files in a folder."""
    extensions = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp')
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and
            f.lower().endswith(extensions)]


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024