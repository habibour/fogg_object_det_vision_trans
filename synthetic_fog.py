"""
Synthetic Fog Generation using Atmospheric Scattering Model (ASM)
Based on the paper "Weather-Aware Object Detection Transformer for Domain Adaptation"

Atmospheric Scattering Model:
    It(x) = Is(x) * e^(-β*d(x)) + A * (1 - e^(-β*d(x)))

Where:
    It(x) = foggy image intensity at pixel x
    Is(x) = clean image intensity at pixel x (source/scene radiance)
    A = atmospheric light (global illumination)
    β = scattering coefficient (controls fog density)
    d(x) = scene depth at pixel x

For images without depth maps, we approximate depth using:
    - Random depth map generation
    - Brightness-based depth estimation
    - Fixed depth map
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import os
from tqdm import tqdm

class FogSimulator:
    """Simulate fog using Atmospheric Scattering Model."""
    
    # Fog density levels based on paper (Low, Mid, High)
    FOG_LEVELS = {
        'low': {'beta': 0.08, 'A': 0.85},       # Light fog
        'mid': {'beta': 0.12, 'A': 0.90},       # Medium fog
        'high': {'beta': 0.16, 'A': 0.95},      # Dense fog
    }
    
    def __init__(self, fog_level='mid', atmospheric_light=None, scattering_coeff=None):
        """
        Initialize fog simulator.
        
        Args:
            fog_level: 'low', 'mid', or 'high'
            atmospheric_light: Override atmospheric light A (0-1)
            scattering_coeff: Override scattering coefficient β
        """
        if fog_level not in self.FOG_LEVELS:
            raise ValueError(f"fog_level must be one of {list(self.FOG_LEVELS.keys())}")
        
        params = self.FOG_LEVELS[fog_level]
        self.beta = scattering_coeff if scattering_coeff is not None else params['beta']
        self.A = atmospheric_light if atmospheric_light is not None else params['A']
        self.fog_level = fog_level
        
    def generate_depth_map(self, image: np.ndarray, method='perlin') -> np.ndarray:
        """
        Generate synthetic depth map for an image.
        
        Args:
            image: Input image (H, W, C)
            method: Depth generation method - 'perlin', 'brightness', 'gradient', or 'random'
            
        Returns:
            depth_map: Depth map normalized to [0, 1] where 0=near, 1=far
        """
        h, w = image.shape[:2]
        
        if method == 'random':
            # Simple random depth with some spatial coherence
            depth = np.random.rand(h // 8, w // 8)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
        elif method == 'brightness':
            # Estimate depth from brightness (darker = closer, brighter = farther)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            depth = gray.astype(np.float32) / 255.0
            
        elif method == 'gradient':
            # Create depth gradient (top=far, bottom=near)
            depth = np.linspace(0.3, 1.0, h).reshape(-1, 1).repeat(w, axis=1)
            # Add some random variation
            noise = np.random.randn(h, w) * 0.1
            depth = np.clip(depth + noise, 0, 1)
            
        elif method == 'perlin':
            # Simple Perlin-like noise using multiple scales
            depth = np.zeros((h, w))
            for scale in [16, 32, 64]:
                h_scaled, w_scaled = h // scale, w // scale
                noise = np.random.rand(h_scaled, w_scaled)
                noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
                depth += noise_resized / scale
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            
        else:
            raise ValueError(f"Unknown depth generation method: {method}")
        
        # Smooth the depth map
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def add_fog(self, 
                image: np.ndarray, 
                depth_map: Optional[np.ndarray] = None,
                depth_method: str = 'perlin') -> Tuple[np.ndarray, np.ndarray]:
        """
        Add synthetic fog to an image using the Atmospheric Scattering Model.
        
        Args:
            image: Input clean image (H, W, 3) in BGR format, uint8
            depth_map: Optional depth map (H, W), values in [0, 1]
            depth_method: Method to generate depth if depth_map is None
            
        Returns:
            foggy_image: Output foggy image (H, W, 3), uint8
            depth_map: Used depth map (H, W), float32
        """
        # Convert image to float [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Generate or validate depth map
        if depth_map is None:
            depth_map = self.generate_depth_map(image, method=depth_method)
        else:
            # Ensure depth map is the same size as image
            if depth_map.shape[:2] != image.shape[:2]:
                depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
            depth_map = depth_map.astype(np.float32)
        
        # Ensure depth map is normalized to [0, 1]
        if depth_map.max() > 1.0:
            depth_map = depth_map / depth_map.max()
        
        # Calculate transmission map: t(x) = e^(-β * d(x))
        transmission = np.exp(-self.beta * depth_map)
        
        # Expand transmission to match image channels
        if len(image_float.shape) == 3:
            transmission = np.expand_dims(transmission, axis=2)
            transmission = np.repeat(transmission, 3, axis=2)
        
        # Apply Atmospheric Scattering Model:
        # It(x) = Is(x) * t(x) + A * (1 - t(x))
        foggy_image = image_float * transmission + self.A * (1 - transmission)
        
        # Clip to valid range and convert back to uint8
        foggy_image = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)
        
        return foggy_image, depth_map
    
    def process_image_file(self, 
                          input_path: str, 
                          output_path: str,
                          save_depth: bool = False) -> None:
        """
        Process a single image file and save the foggy version.
        
        Args:
            input_path: Path to input clean image
            output_path: Path to save foggy image
            save_depth: If True, also save the depth map
        """
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Add fog
        foggy_image, depth_map = self.add_fog(image)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save foggy image
        cv2.imwrite(output_path, foggy_image)
        
        # Optionally save depth map
        if save_depth:
            depth_path = output_path.replace('.jpg', '_depth.png').replace('.png', '_depth.png')
            depth_vis = (depth_map * 255).astype(np.uint8)
            cv2.imwrite(depth_path, depth_vis)

def process_dataset(input_dir: str, 
                    output_dir: str, 
                    fog_levels: list = ['low', 'mid', 'high'],
                    save_depth: bool = False) -> None:
    """
    Process entire dataset to generate foggy versions.
    
    Args:
        input_dir: Directory containing clean images
        output_dir: Directory to save foggy images
        fog_levels: List of fog levels to generate
        save_depth: Whether to save depth maps
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'**/*{ext}'))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each fog level
    for fog_level in fog_levels:
        print(f"\nGenerating {fog_level} fog...")
        simulator = FogSimulator(fog_level=fog_level)
        
        # Create output directory for this fog level
        fog_output_dir = output_path / fog_level
        fog_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_file in tqdm(image_files):
            try:
                # Maintain directory structure
                rel_path = img_file.relative_to(input_path)
                output_file = fog_output_dir / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Process image
                simulator.process_image_file(
                    str(img_file), 
                    str(output_file),
                    save_depth=save_depth
                )
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

def visualize_fog_levels(image_path: str, output_dir: str = 'fog_examples'):
    """
    Generate and visualize all fog levels for a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate fog at different levels
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original (Clean)')
    axes[0, 0].axis('off')
    
    # Generate each fog level
    for idx, fog_level in enumerate(['low', 'mid', 'high']):
        simulator = FogSimulator(fog_level=fog_level)
        foggy_image, depth_map = simulator.add_fog(image)
        foggy_rgb = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB)
        
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        axes[row, col].imshow(foggy_rgb)
        axes[row, col].set_title(f'{fog_level.capitalize()} Fog (β={simulator.beta:.2f})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fog_comparison.png', dpi=150)
    print(f"Visualization saved to {output_dir}/fog_comparison.png")
    plt.close()

if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Synthetic Fog Generation using ASM")
    print("=" * 60)
    
    # Example 1: Process single image
    # test_image = 'voc_2012/VOC2012_filtered/JPEGImages/2008_000008.jpg'
    # if os.path.exists(test_image):
    #     visualize_fog_levels(test_image)
    
    # Example 2: Process entire dataset
    INPUT_DIR = 'voc_2012/VOC2012_filtered/JPEGImages'
    OUTPUT_DIR = 'voc_2012/VOC2012_foggy'
    
    if os.path.exists(INPUT_DIR):
        print(f"\nProcessing dataset: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        process_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            fog_levels=['low', 'mid', 'high'],
            save_depth=True  # Save depth maps for analysis
        )
        
        print("\n" + "=" * 60)
        print("Fog generation complete!")
        print("=" * 60)
    else:
        print(f"Input directory not found: {INPUT_DIR}")
        print("Please run filter_voc_classes.py first to create filtered dataset")
