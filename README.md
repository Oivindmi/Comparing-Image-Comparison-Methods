# Image Matching Methods Comparison

This tool helps you evaluate different image matching methods to find the best approach for identifying similar images across different cameras.

## Features

- Test multiple image matching methods:
  - **CLIP** (Contrastive Language-Image Pre-training): OpenAI's robust visual embedding model
  - **DINO/DINOv2**: Facebook's self-supervised feature extraction model
  - **VGG Features**: Classic CNN-based features for image matching
  - **ORB** (Oriented FAST and Rotated BRIEF): Traditional computer vision feature matching
  - **SSIM** (Structural Similarity Index): Direct pixel-level comparison
  - **ImageBind**: Facebook's multi-modal embedding model that works well for scene understanding

- Compare performance metrics:
  - Processing time
  - Memory usage
  - Matching quality

- Interactive GUI to explore matching results

## Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required dependencies:
   ```
   pip install torch torchvision torchaudio
   pip install opencv-python scikit-image matplotlib tqdm pillow psutil numpy
   pip install git+https://github.com/openai/CLIP.git
   ```

3. Additional dependencies will be installed automatically when needed

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. Select your folders:
   - Folder 1: Images from the first camera
   - Folder 2: Images from the second camera

3. Choose which methods to test

4. Set the number of top matches to display

5. Click "Run Comparison" to start the analysis

6. Review the results in the tabs for each method and the performance comparison

## Project Structure

- `main.py`: The main application with GUI
- `methods/`: Implementations of different matching algorithms
  - `clip_matcher.py`: CLIP-based matching
  - `dino_matcher.py`: DINO/DINOv2-based matching
  - `vgg_matcher.py`: VGG feature-based matching
  - `orb_matcher.py`: ORB feature matching
  - `ssim_matcher.py`: SSIM-based matching
  - `imagebind_matcher.py`: ImageBind-based matching

## System Requirements

This tool has been optimized for macOS with Apple Silicon (M1/M2/M3) but works on any system with:
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for larger image sets)
- Modern CPU (GPU optional but recommended)

On Apple Silicon Macs, the tool will automatically use the Metal Performance Shaders (MPS) backend for acceleration.

## Tips for Best Results

- For consistent evaluation, use images of similar resolution across both folders
- Start with a smaller set of images (20-30 per folder) for initial testing
- The "Performance Comparison" tab helps identify which methods work best for your specific images
- For production use, choose the method with the best balance of accuracy and performance