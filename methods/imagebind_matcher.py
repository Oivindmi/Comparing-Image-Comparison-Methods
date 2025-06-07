import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import gc
import sys
import importlib
import time
import cv2
from pathlib import Path

# Suppress OpenCV warnings
cv2.setLogLevel(0)

# Ensure ImageBind is in path
imagebind_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imagebind")
if imagebind_dir not in sys.path:
    sys.path.insert(0, imagebind_dir)


def safe_image_open(image_path):
    """
    Safely open an image file handling Unicode paths and various formats.
    """
    try:
        # Method 1: Try PIL directly (works well with Unicode)
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as pil_error:
        try:
            # Method 2: Try using pathlib for better Unicode handling
            path_obj = Path(image_path)
            if path_obj.exists():
                image = Image.open(path_obj).convert("RGB")
                return image
        except Exception as pathlib_error:
            try:
                # Method 3: Try OpenCV with Unicode support then convert to PIL
                # Use cv2.IMREAD_COLOR and handle Unicode paths
                import cv2

                # For Windows Unicode support, read as binary first
                if os.name == 'nt':  # Windows
                    # Use numpy to read file with Unicode support
                    with open(image_path, 'rb') as f:
                        image_data = f.read()

                    # Convert to numpy array and decode with OpenCV
                    nparr = np.frombuffer(image_data, np.uint8)
                    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if cv_image is not None:
                        # Convert BGR to RGB and then to PIL
                        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(cv_image_rgb)
                        return image
                else:
                    # For non-Windows systems
                    cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if cv_image is not None:
                        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(cv_image_rgb)
                        return image

            except Exception as cv_error:
                # Method 4: Try with different encoding
                try:
                    # Sometimes encoding the path differently helps
                    encoded_path = image_path.encode('utf-8').decode('utf-8')
                    image = Image.open(encoded_path).convert("RGB")
                    return image
                except Exception as encoding_error:
                    # Log the specific errors for debugging
                    print(f"Failed to open {image_path}:")
                    print(f"  PIL error: {pil_error}")
                    print(f"  Pathlib error: {pathlib_error}")
                    print(f"  OpenCV error: {cv_error}")
                    print(f"  Encoding error: {encoding_error}")
                    return None

    return None


def get_image_files_unicode_safe(folder):
    """Get all image files in a folder with proper Unicode handling."""
    extensions = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.bmp', '.tiff', '.tif')
    image_files = []

    try:
        # Use pathlib for better Unicode support
        folder_path = Path(folder)

        if not folder_path.exists():
            print(f"Folder does not exist: {folder}")
            return []

        # Recursively find all image files
        for ext in extensions:
            # Use both cases for extensions
            image_files.extend(folder_path.glob(f"**/*{ext}"))
            image_files.extend(folder_path.glob(f"**/*{ext.upper()}"))

        # Convert back to strings and remove duplicates
        image_files = list(set([str(f) for f in image_files if f.is_file()]))

        print(f"Found {len(image_files)} image files in {folder}")
        return image_files

    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
        # Fallback to original method
        try:
            return [os.path.join(folder, f) for f in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, f)) and
                    f.lower().endswith(extensions)]
        except Exception as fallback_error:
            print(f"Fallback method also failed: {fallback_error}")
            return []


def match_images(folder1, folder2, top_n=5):
    """
    Match images between two folders using ImageBind embeddings.
    This version includes Unicode support and optimizations for speed.

    Args:
        folder1: Path to first folder containing images
        folder2: Path to second folder containing images
        top_n: Number of top matches to return

    Returns:
        Dictionary containing matches, processing time, and memory usage
    """
    start_time = time.time()
    start_memory = get_memory_usage()

    # Check for best available device (prioritizing GPU)
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # Get GPU memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        reserved_mem = torch.cuda.memory_reserved(0) / 1024 ** 3
        allocated_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
        free_mem = total_mem - allocated_mem
        print(f"GPU Memory: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total ({free_mem:.2f}GB free)")
    elif hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps,
                                                    'is_available') and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU - no compatible GPU detected")

    # Try to import ImageBind
    try:
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType

        print("Successfully imported ImageBind!")

        # Load ImageBind model
        model = imagebind_model.imagebind_huge(pretrained=True)
        model = model.to(device)
        model.eval()

        # OPTIMIZATION: Use half-precision if on GPU for 2x memory savings and speed
        if device == "cuda":
            print("Enabling half-precision (FP16) for faster inference")
            model = model.half()

        # Define image transformation for ImageBind
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # OPTIMIZATION: Function to process images in batches with Unicode support
        def extract_features_batch(image_paths, batch_size=32):
            """Process images in batches for faster inference with Unicode support"""
            features = {}

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                valid_images = []
                valid_paths = []

                # Preprocess all images in the batch with safe image loading
                for img_path in batch_paths:
                    try:
                        # Use safe image loading for Unicode support
                        image = safe_image_open(img_path)

                        if image is None:
                            continue

                        # Skip small images
                        if image.size[0] < 10 or image.size[1] < 10:
                            continue

                        # Process image
                        img_tensor = preprocess(image)

                        # Track valid images and their paths
                        valid_images.append(img_tensor)
                        valid_paths.append(img_path)
                    except Exception as e:
                        print(f"Batch processing error for {img_path}: {e}")
                        continue

                if not valid_images:
                    continue

                # Stack valid images to create batch tensor
                try:
                    # Efficiently move data to GPU
                    with torch.no_grad():
                        # Stack all valid images
                        batch_tensor = torch.stack(valid_images).to(device)

                        # Convert to half precision if using CUDA
                        if device == "cuda":
                            batch_tensor = batch_tensor.half()

                        # Process batch through model
                        inputs = {ModalityType.VISION: batch_tensor}
                        embeddings = model(inputs)

                        # Store embeddings for each valid image
                        for idx, img_path in enumerate(valid_paths):
                            features[img_path] = embeddings[ModalityType.VISION][idx].cpu().numpy()
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Fallback to individual processing for this batch
                    for img_path in valid_paths:
                        try:
                            feat = extract_features(img_path)
                            if feat is not None:
                                features[img_path] = feat
                        except Exception:
                            pass

            return features

        # Function for sequential processing (fallback) with Unicode support
        def extract_features(image_path):
            try:
                # Use safe image loading
                image = safe_image_open(image_path)

                if image is None:
                    return None

                # Check image validity
                if image.size[0] < 10 or image.size[1] < 10:
                    return None

                # Preprocess image
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                # Convert to half precision if using CUDA
                if device == "cuda":
                    image_tensor = image_tensor.half()

                # Create inputs for ImageBind using ModalityType.VISION
                inputs = {ModalityType.VISION: image_tensor}

                with torch.no_grad():
                    embeddings = model(inputs)

                return embeddings[ModalityType.VISION][0].cpu().numpy()

            except Exception as e:
                print(f"Individual processing error for {image_path}: {e}")
                return None

    except Exception as e:
        print(f"Could not load ImageBind: {e}")
        print("Using a fallback model: CLIP...")

        # Fallback to CLIP if ImageBind fails
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.to(device)
        model.eval()

        # OPTIMIZATION: Use half-precision if on GPU
        if device == "cuda":
            model = model.half()

        # Function for batch processing with CLIP and Unicode support
        def extract_features_batch(image_paths, batch_size=64):
            features = {}
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                valid_images = []
                valid_paths = []

                for img_path in batch_paths:
                    try:
                        image = safe_image_open(img_path)
                        if image is None:
                            continue
                        img_tensor = preprocess(image)
                        valid_images.append(img_tensor)
                        valid_paths.append(img_path)
                    except Exception:
                        pass

                if not valid_images:
                    continue

                try:
                    # Process batch through CLIP
                    batch_tensor = torch.stack(valid_images).to(device)
                    if device == "cuda":
                        batch_tensor = batch_tensor.half()

                    with torch.no_grad():
                        batch_features = model.encode_image(batch_tensor)

                    # Store embeddings
                    for idx, img_path in enumerate(valid_paths):
                        features[img_path] = batch_features[idx].cpu().numpy()
                except Exception as e:
                    print(f"Error processing CLIP batch: {e}")

            return features

        # Sequential version for CLIP (fallback) with Unicode support
        def extract_features(image_path):
            try:
                image = safe_image_open(image_path)
                if image is None:
                    return None

                image_tensor = preprocess(image).unsqueeze(0).to(device)

                if device == "cuda":
                    image_tensor = image_tensor.half()

                with torch.no_grad():
                    features = model.encode_image(image_tensor)

                return features[0].cpu().numpy()
            except Exception:
                return None

    # Get image files with Unicode support
    img_files1 = get_image_files_unicode_safe(folder1)
    img_files2 = get_image_files_unicode_safe(folder2)

    print(f"Processing {len(img_files1)} images from folder 1")
    print(f"Processing {len(img_files2)} images from folder 2")

    # OPTIMIZATION: Determine optimal batch size based on GPU memory
    if device == "cuda":
        available_mem = free_mem * 0.8  # Use 80% of free memory
        # Estimate memory per image (in GB)
        mem_per_image = 0.01  # ~10MB per image is a safe estimate
        optimal_batch_size = max(4, min(32, int(available_mem / mem_per_image)))
        print(f"Using batch size of {optimal_batch_size} based on available GPU memory")
    else:
        optimal_batch_size = 16  # Smaller batch size for CPU

    # Extract features for both folders using batched processing
    print("Extracting features from folder 1...")
    features1 = extract_features_batch(img_files1, batch_size=optimal_batch_size)

    print("Extracting features from folder 2...")
    features2 = extract_features_batch(img_files2, batch_size=optimal_batch_size)

    print(f"Successfully processed {len(features1)} images from folder 1")
    print(f"Successfully processed {len(features2)} images from folder 2")

    # OPTIMIZATION: Compute similarities with vectorized operations
    print("Computing similarities...")
    # Convert features to matrices for efficient computation
    paths1 = list(features1.keys())
    paths2 = list(features2.keys())

    if paths1 and paths2:
        # Stack features into matrices
        feat_matrix1 = np.vstack([features1[p] for p in paths1])
        feat_matrix2 = np.vstack([features2[p] for p in paths2])

        # Compute all similarities at once
        similarity_matrix = cosine_similarity(feat_matrix1, feat_matrix2)

        # Convert to list of (path1, path2, similarity) tuples
        similarities = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                similarities.append((paths1[i], paths2[j], similarity_matrix[i, j]))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)

        # Take top N matches
        top_matches = similarities[:top_n]
    else:
        top_matches = []

    # Clean up to free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    end_time = time.time()
    processing_time = end_time - start_time
    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory

    print(f"Processing complete! Time taken: {processing_time:.2f}s")

    return {
        "matches": top_matches,
        "processing_time": processing_time,
        "memory_usage": memory_used
    }


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024