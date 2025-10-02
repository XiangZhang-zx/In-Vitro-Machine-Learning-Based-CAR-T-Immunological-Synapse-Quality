#!/usr/bin/env python3
"""
Train a custom cellpose model on the converted dataset.
"""

import os
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from cellpose import models, io, train
from cellpose.io import logger_setup
import logging


def setup_logging(log_level='INFO'):
    """Setup logging for cellpose training."""
    logger_setup()
    logging.getLogger().setLevel(getattr(logging, log_level))


def prepare_training_data(data_dir):
    """
    Prepare training data for cellpose.
    
    Args:
        data_dir: Directory containing train/, val/, test/ folders with images and masks
    
    Returns:
        train_data: List of training data paths
        test_data: List of validation data paths
    """
    data_path = Path(data_dir)
    
    # Get training images and masks
    train_dir = data_path / 'train'
    train_images = sorted([str(f) for f in train_dir.glob('*.png') if not f.name.endswith('_masks.png')])
    train_masks = sorted([str(f) for f in train_dir.glob('*_masks.png')])
    
    # Get validation images and masks
    val_dir = data_path / 'val'
    val_images = sorted([str(f) for f in val_dir.glob('*.png') if not f.name.endswith('_masks.png')])
    val_masks = sorted([str(f) for f in val_dir.glob('*_masks.png')])
    
    print(f"Found {len(train_images)} training images")
    print(f"Found {len(train_masks)} training masks")
    print(f"Found {len(val_images)} validation images")
    print(f"Found {len(val_masks)} validation masks")
    
    # Verify that images and masks match
    assert len(train_images) == len(train_masks), "Mismatch between training images and masks"
    assert len(val_images) == len(val_masks), "Mismatch between validation images and masks"
    
    # Create training data list
    train_data = []
    for img_path, mask_path in zip(train_images, train_masks):
        train_data.append([img_path, mask_path])
    
    # Create validation data list
    test_data = []
    for img_path, mask_path in zip(val_images, val_masks):
        test_data.append([img_path, mask_path])
    
    return train_data, test_data


def train_cellpose_model(train_data, test_data, model_name='custom_cell_model',
                        n_epochs=100, learning_rate=0.1, weight_decay=0.0001,
                        batch_size=8, model_type='cyto2', channels=[0,0]):
    """
    Train a custom cellpose model.

    Args:
        train_data: List of [image_path, mask_path] pairs for training
        test_data: List of [image_path, mask_path] pairs for validation
        model_name: Name for the saved model
        n_epochs: Number of training epochs
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        batch_size: Batch size for training
        model_type: Base model type ('cyto', 'cyto2', 'nuclei', etc.)
        channels: Channel configuration [cytoplasm_channel, nucleus_channel]

    Returns:
        model: Trained cellpose model
    """

    # Train the model using the train module
    print(f"Starting training with {len(train_data)} training samples and {len(test_data)} validation samples")
    print(f"Training parameters:")
    print(f"  - Epochs: {n_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: {channels}")
    print(f"  - Model type: {model_type}")

    # Use the train module for training
    model_path = train.train_seg(
        train_data=train_data,
        test_data=test_data,
        channels=channels,
        save_path='./',
        save_every=10,
        save_each=True,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        model_name=model_name,
        pretrained_model=model_type
    )

    # Load the trained model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    print(f"Model saved to: {model_path}")
    return model, model_path


def evaluate_model(model_path, test_data_dir, output_dir='evaluation_results'):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to the trained model
        test_data_dir: Directory containing test images
        output_dir: Directory to save evaluation results
    """
    
    # Load the trained model
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get test images
    test_dir = Path(test_data_dir) / 'test'
    test_images = sorted([str(f) for f in test_dir.glob('*.png') if not f.name.endswith('_masks.png')])
    
    print(f"Evaluating on {len(test_images)} test images")
    
    # Process each test image
    for i, img_path in enumerate(test_images[:5]):  # Evaluate on first 5 images
        print(f"Processing {Path(img_path).name}")
        
        # Load image
        img = io.imread(img_path)
        
        # Run cellpose
        masks, flows, styles = model.eval(img, diameter=None, channels=[0,0])
        
        # Save results
        img_name = Path(img_path).stem
        
        # Save mask
        mask_save_path = output_path / f"{img_name}_predicted_masks.png"
        io.imsave(str(mask_save_path), masks.astype(np.uint16))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(img.shape) == 3:
            axes[0].imshow(img)
        else:
            axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predicted masks
        axes[1].imshow(masks, cmap='tab20')
        axes[1].set_title(f'Predicted Masks (max ID: {np.max(masks)})')
        axes[1].axis('off')
        
        # Flows
        axes[2].imshow(flows[0])
        axes[2].set_title('Flow Field')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        vis_save_path = output_path / f"{img_name}_evaluation.png"
        plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {mask_save_path}")
        print(f"  Saved: {vis_save_path}")
        print(f"  Found {np.max(masks)} cells")


def main():
    parser = argparse.ArgumentParser(description='Train cellpose model on custom dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/cellpose_format',
                        help='Directory containing train/, val/, test/ folders')
    parser.add_argument('--model_name', type=str, default='custom_cell_model',
                        help='Name for the saved model')
    parser.add_argument('--model_type', type=str, default='cyto2',
                        choices=['cyto', 'cyto2', 'nuclei', 'tissuenet', 'livecell', 'cyto3'],
                        help='Base model type to start from')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for regularization')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 0],
                        help='Channel configuration [cytoplasm_channel, nucleus_channel]')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate existing model')
    parser.add_argument('--model_path', type=str,
                        help='Path to existing model for evaluation')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.evaluate_only:
        if not args.model_path:
            print("Error: --model_path required for evaluation")
            return
        
        print("Evaluating existing model...")
        evaluate_model(args.model_path, args.data_dir)
    
    else:
        # Prepare training data
        print("Preparing training data...")
        train_data, test_data = prepare_training_data(args.data_dir)
        
        # Train model
        print("Training model...")
        model, model_path = train_cellpose_model(
            train_data=train_data,
            test_data=test_data,
            model_name=args.model_name,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            model_type=args.model_type,
            channels=args.channels
        )
        
        # Evaluate model
        print("Evaluating trained model...")
        evaluate_model(model_path, args.data_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
