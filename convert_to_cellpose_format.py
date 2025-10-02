#!/usr/bin/env python3
"""
Convert annotation images to cellpose format.

Cellpose requires:
1. Original images (RGB or grayscale)
2. Mask images where each cell has a unique integer ID (1, 2, 3, ...)
3. Background pixels should be 0
4. Masks should be saved as 16-bit PNG files with suffix '_masks.png'
"""

import os
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from scipy import ndimage
import matplotlib.pyplot as plt


def analyze_annotation_image(img_path):
    """Analyze the annotation image to understand its format."""
    print(f"Analyzing: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {img_rgb.shape}")
    print(f"Image dtype: {img_rgb.dtype}")
    print(f"Value range: {img_rgb.min()} - {img_rgb.max()}")
    
    # Get unique colors
    unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
    print(f"Number of unique colors: {len(unique_colors)}")
    print("First 10 unique colors:")
    for i, color in enumerate(unique_colors[:10]):
        print(f"  Color {i}: RGB{tuple(color)}")
    
    return img_rgb, unique_colors


def color_to_mask(img_rgb, method='connected_components'):
    """
    Convert colored annotation to cellpose mask format.
    
    Args:
        img_rgb: RGB annotation image
        method: 'connected_components' or 'color_clustering'
    
    Returns:
        mask: 2D array with unique cell IDs
    """
    if method == 'connected_components':
        # Convert to grayscale for connected components
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Threshold to create binary image (assuming non-black pixels are cells)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        # Convert to cellpose format (background=0, cells=1,2,3,...)
        mask = labels.astype(np.uint16)
        
        print(f"Found {num_labels-1} connected components")
        
    elif method == 'color_clustering':
        # Reshape image for clustering
        pixels = img_rgb.reshape(-1, 3)
        
        # Find background (most common color, usually black)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        background_color = unique_colors[np.argmax(counts)]
        
        # Create mask for non-background pixels
        non_bg_mask = ~np.all(img_rgb == background_color, axis=2)
        
        if np.sum(non_bg_mask) == 0:
            print("Warning: No non-background pixels found")
            return np.zeros(img_rgb.shape[:2], dtype=np.uint16)
        
        # Get non-background pixels
        non_bg_pixels = img_rgb[non_bg_mask]
        
        # Cluster colors to identify different cells
        n_clusters = min(len(np.unique(non_bg_pixels.reshape(-1, 3), axis=0)), 50)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(non_bg_pixels)
            
            # Create mask
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint16)
            mask[non_bg_mask] = cluster_labels + 1  # +1 because background is 0
            
            # Apply connected components to each cluster to separate touching cells
            final_mask = np.zeros_like(mask)
            current_id = 1
            
            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = (mask == cluster_id).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(cluster_mask)
                
                for label_id in range(1, num_labels):
                    final_mask[labels == label_id] = current_id
                    current_id += 1
            
            mask = final_mask
        else:
            # Single cluster, use connected components
            binary = non_bg_mask.astype(np.uint8) * 255
            num_labels, labels = cv2.connectedComponents(binary)
            mask = labels.astype(np.uint16)
        
        print(f"Found {np.max(mask)} cells using color clustering")
    
    return mask


def convert_dataset(input_dir, output_dir, annotation_suffix='_annotation', method='connected_components'):
    """
    Convert entire dataset to cellpose format.
    
    Args:
        input_dir: Directory containing train/ and train_annotation/ folders
        output_dir: Output directory for cellpose format
        annotation_suffix: Suffix for annotation folders
        method: Method for mask conversion
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(exist_ok=True)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        img_dir = input_path / split
        ann_dir = input_path / f"{split}{annotation_suffix}"
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"Skipping {split}: directories not found")
            continue
        
        output_split_dir = output_path / split
        output_split_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        
        # Get list of images
        img_files = list(img_dir.glob("*.png"))
        
        for img_file in img_files:
            ann_file = ann_dir / img_file.name
            
            if not ann_file.exists():
                print(f"Warning: No annotation found for {img_file.name}")
                continue
            
            print(f"Processing: {img_file.name}")
            
            # Copy original image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Error: Could not load {img_file}")
                continue
            
            output_img_path = output_split_dir / img_file.name
            cv2.imwrite(str(output_img_path), img)
            
            # Convert annotation to mask
            ann_img = cv2.imread(str(ann_file))
            if ann_img is None:
                print(f"Error: Could not load {ann_file}")
                continue
            
            ann_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
            mask = color_to_mask(ann_rgb, method=method)
            
            # Save mask with cellpose naming convention
            mask_name = img_file.stem + '_masks.png'
            mask_path = output_split_dir / mask_name
            
            # Save as 16-bit PNG
            cv2.imwrite(str(mask_path), mask.astype(np.uint16))
            
            print(f"  Saved: {output_img_path}")
            print(f"  Saved: {mask_path} (max ID: {np.max(mask)})")


def visualize_conversion(original_img_path, annotation_img_path, output_dir):
    """Visualize the conversion process for debugging."""
    # Load images
    original = cv2.imread(original_img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    annotation = cv2.imread(annotation_img_path)
    annotation_rgb = cv2.cvtColor(annotation, cv2.COLOR_BGR2RGB)
    
    # Convert to mask
    mask = color_to_mask(annotation_rgb, method='connected_components')
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(annotation_rgb)
    axes[1].set_title('Annotation')
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='tab20')
    axes[2].set_title(f'Cellpose Mask (max ID: {np.max(mask)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    vis_path = Path(output_dir) / 'conversion_visualization.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert annotations to cellpose format')
    parser.add_argument('--input_dir', type=str, default='dataset/original',
                        help='Input directory containing train/ and train_annotation/ folders')
    parser.add_argument('--output_dir', type=str, default='dataset/cellpose_format',
                        help='Output directory for cellpose format')
    parser.add_argument('--method', type=str, choices=['connected_components', 'color_clustering'],
                        default='connected_components', help='Method for mask conversion')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze the first annotation image')
    parser.add_argument('--visualize', type=str, nargs=2, metavar=('ORIGINAL', 'ANNOTATION'),
                        help='Visualize conversion for specific image pair')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Analyze first annotation image
        ann_dir = Path(args.input_dir) / 'train_annotation'
        ann_files = list(ann_dir.glob("*.png"))
        if ann_files:
            analyze_annotation_image(str(ann_files[0]))
        else:
            print("No annotation files found")
    
    elif args.visualize:
        visualize_conversion(args.visualize[0], args.visualize[1], args.output_dir)
    
    else:
        # Convert entire dataset
        convert_dataset(args.input_dir, args.output_dir, method=args.method)
        print(f"\nConversion complete! Cellpose format data saved to: {args.output_dir}")
