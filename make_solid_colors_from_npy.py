#!/usr/bin/env python3
"""
Generate solid color masks from Cellpose .npy mask files
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from matplotlib.colors import hsv_to_rgb

def generate_solid_color_mask(mask_array):
    """
    Convert a Cellpose mask array (with cell IDs) to a solid color RGB image
    
    Args:
        mask_array: 2D numpy array where each cell has a unique integer ID
    
    Returns:
        RGB image with each cell as a solid color
    """
    h, w = mask_array.shape
    output_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get unique cell IDs (excluding 0 which is background)
    cell_ids = np.unique(mask_array)
    cell_ids = cell_ids[cell_ids > 0]
    
    # Assign a solid color to each cell
    for cell_id in cell_ids:
        # Create mask for this cell
        cell_mask = mask_array == cell_id
        
        # Generate a bright, saturated color based on cell ID
        np.random.seed(int(cell_id) + 42)  # Consistent colors
        hue = np.random.rand()
        
        # Convert HSV to RGB (full saturation and value for bright colors)
        solid_color = hsv_to_rgb([hue, 1.0, 1.0])
        solid_color = (solid_color * 255).astype(np.uint8)
        
        # Fill this cell with solid color
        output_img[cell_mask] = solid_color
    
    return output_img

def process_masks_directory(input_dir, output_dir):
    """Process all .npy mask files in a directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all _masks.npy files
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('_masks.npy')]
    
    print(f"Found {len(mask_files)} mask files to process")
    
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        try:
            # Load mask array
            mask_path = os.path.join(input_dir, mask_file)
            mask_array = np.load(mask_path, allow_pickle=True)
            
            # Generate solid color image
            solid_color_img = generate_solid_color_mask(mask_array)
            
            # Create output filename
            output_filename = mask_file.replace('_masks.npy', '_solid_colors.png')
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as PNG
            Image.fromarray(solid_color_img).save(output_path)
            
            # Count cells
            num_cells = len(np.unique(mask_array)) - 1  # Exclude background
            
            if num_cells > 0:
                print(f"\n{mask_file}: {num_cells} cells")
            
        except Exception as e:
            print(f"\nError processing {mask_file}: {str(e)}")
            continue
    
    print(f"\nProcessing completed! Output saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate solid color masks from Cellpose .npy files')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory with _masks.npy files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for solid color PNG images')
    
    args = parser.parse_args()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    process_masks_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
