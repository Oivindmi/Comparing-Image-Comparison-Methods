import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
import gc
import torchvision.transforms as T


def match_images(folder1, folder2, top_n=5):
    """
    Match images between two folders using DINOv2 embeddings.

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

    # Load DINOv2 model - using the smallest version for efficiency
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    except Exception as e:
        print(f"Error loading DINOv2 from torch.hub: {e}")
        print("Attempting to load pre-trained DINOv2...")
        # Fallback to manual loading
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

    model = model.to(device)
    model.eval()

    # Define image transformation
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get image files
    img_files1 = get_image_files(folder1)
    img_files2 = get_image_files(folder2)

    print(f"Processing {len(img_files1)} images from folder 1")
    print(f"Processing {len(img_files2)} images from folder 2")

    # Extract features for folder 1
    features1 = {}
    for img_path in tqdm(img_files1, desc="Extracting features from folder 1"):
        try:
            feat = extract_features(img_path, model, transform, device)
            if feat is not None:
                features1[img_path] = feat
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Extract features for folder 2
    features2 = {}
    for img_path in tqdm(img_files2, desc="Extracting features from folder 2"):
        try:
            feat = extract_features(img_path, model, transform, device)
            if feat is not None:
                features2[img_path] = feat
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Compute similarities between all pairs
    similarities = []
    for path1, feat1 in tqdm(features1.items(), desc="Computing similarities"):
        for path2, feat2 in features2.items():
            sim = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0][0]
            similarities.append((path1, path2, sim))

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


def extract_features(image_path, model, transform, device):
    """Extract image features using DINOv2."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.cpu().numpy()
    except Exception as e:
        print(f"Error in extract_features for {image_path}: {e}")
        return None


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