import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import gc


def match_images(folder1, folder2, top_n=5):
    """
    Match images between two folders using ORB features and Brute Force matching.
    This is a traditional computer vision approach without deep learning.

    Args:
        folder1: Path to first folder containing images
        folder2: Path to second folder containing images
        top_n: Number of top matches to return

    Returns:
        Dictionary containing matches, processing time, and memory usage
    """
    start_memory = get_memory_usage()

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Get image files
    img_files1 = get_image_files(folder1)
    img_files2 = get_image_files(folder2)

    print(f"Processing {len(img_files1)} images from folder 1")
    print(f"Processing {len(img_files2)} images from folder 2")

    # Extract features for folder 1
    features1 = {}
    for img_path in tqdm(img_files1, desc="Extracting ORB features from folder 1"):
        try:
            keypoints, descriptors = extract_orb_features(img_path, orb)
            if descriptors is not None:
                features1[img_path] = (keypoints, descriptors)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Extract features for folder 2
    features2 = {}
    for img_path in tqdm(img_files2, desc="Extracting ORB features from folder 2"):
        try:
            keypoints, descriptors = extract_orb_features(img_path, orb)
            if descriptors is not None:
                features2[img_path] = (keypoints, descriptors)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Create Brute Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Compute similarities between all pairs
    similarities = []
    for path1, (kp1, desc1) in tqdm(features1.items(), desc="Computing ORB matches"):
        if desc1 is None:
            continue

        for path2, (kp2, desc2) in features2.items():
            if desc2 is None:
                continue

            try:
                # Match descriptors
                matches = bf.match(desc1, desc2)

                # Sort them in order of distance (lower is better)
                matches = sorted(matches, key=lambda x: x.distance)

                # Calculate a similarity score based on match quality
                if len(matches) > 0:
                    # Use number of good matches divided by average distance as similarity
                    # (higher is better)
                    good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches

                    if len(good_matches) > 5:  # Require at least 5 good matches
                        avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                        match_ratio = len(good_matches) / max(len(kp1), len(kp2))
                        # Similarity score: higher is better
                        similarity = match_ratio * (100 / (avg_distance + 1e-5))
                        similarities.append((path1, path2, similarity))
            except Exception as e:
                print(f"Error matching {path1} and {path2}: {e}")

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Take top N matches
    top_matches = similarities[:top_n]

    # Clean up to free memory
    del features1
    del features2
    gc.collect()

    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory

    return {
        "matches": top_matches,
        "memory_usage": memory_used
    }


def extract_orb_features(image_path, orb):
    """Extract ORB features from an image."""
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL if OpenCV fails (e.g., for HEIC files)
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if too large (for consistency and memory efficiency)
        max_dim = 1000
        h, w = gray.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        return keypoints, descriptors

    except Exception as e:
        print(f"Error in extract_orb_features for {image_path}: {e}")
        return None, None


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