import os
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import gc


def match_images(folder1, folder2, top_n=5):
    """
    Match images between two folders using Structural Similarity Index (SSIM).
    This is a direct pixel-based comparison that works well for similar viewpoints.

    Args:
        folder1: Path to first folder containing images
        folder2: Path to second folder containing images
        top_n: Number of top matches to return

    Returns:
        Dictionary containing matches, processing time, and memory usage
    """
    start_memory = get_memory_usage()

    # Get image files
    img_files1 = get_image_files(folder1)
    img_files2 = get_image_files(folder2)

    print(f"Processing {len(img_files1)} images from folder 1")
    print(f"Processing {len(img_files2)} images from folder 2")

    # Process images
    processed_images1 = {}
    for img_path in tqdm(img_files1, desc="Processing images from folder 1"):
        try:
            img = preprocess_image(img_path)
            if img is not None:
                processed_images1[img_path] = img
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    processed_images2 = {}
    for img_path in tqdm(img_files2, desc="Processing images from folder 2"):
        try:
            img = preprocess_image(img_path)
            if img is not None:
                processed_images2[img_path] = img
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Compute SSIM between all pairs
    similarities = []
    for path1, img1 in tqdm(processed_images1.items(), desc="Computing SSIM"):
        for path2, img2 in processed_images2.items():
            try:
                # Calculate SSIM
                score = compute_ssim(img1, img2)
                similarities.append((path1, path2, score))
            except Exception as e:
                print(f"Error computing SSIM for {path1} and {path2}: {e}")

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Take top N matches
    top_matches = similarities[:top_n]

    # Clean up to free memory
    del processed_images1
    del processed_images2
    gc.collect()

    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory

    return {
        "matches": top_matches,
        "memory_usage": memory_used
    }


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for SSIM comparison."""
    try:
        # Try with OpenCV first
        img = cv2.imread(image_path)

        if img is None:
            # Try with PIL if OpenCV fails (e.g., for HEIC files)
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        resized = cv2.resize(gray, target_size)

        return resized

    except Exception as e:
        print(f"Error in preprocess_image for {image_path}: {e}")
        return None


def compute_ssim(img1, img2):
    """Compute Structural Similarity Index between two images."""
    try:
        # Ensure images are the same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM
        score, _ = ssim(img1, img2, full=True)
        return score

    except Exception as e:
        print(f"Error in compute_ssim: {e}")
        return 0.0


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