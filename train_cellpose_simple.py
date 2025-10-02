#!/usr/bin/env python3
"""
Simple cellpose training script using the command line interface.
"""

import os
import subprocess
import argparse
from pathlib import Path


def run_cellpose_training(data_dir, model_name='custom_cell_model', n_epochs=100, 
                         learning_rate=0.1, batch_size=8):
    """
    Run cellpose training using command line interface.
    
    Args:
        data_dir: Directory containing train/, val/ folders
        model_name: Name for the saved model
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
    """
    
    data_path = Path(data_dir)
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    # Build cellpose command
    cmd = [
        'python', '-m', 'cellpose',
        '--train',
        '--dir', str(train_dir),
        '--test_dir', str(val_dir),
        '--pretrained_model', 'cyto2',
        '--chan', '0',
        '--chan2', '0',
        '--learning_rate', str(learning_rate),
        '--n_epochs', str(n_epochs),
        '--batch_size', str(batch_size),
        '--save_every', '10',
        '--save_each',
        '--model_name', model_name,
        '--verbose'
    ]
    
    print("Running cellpose training with command:")
    print(' '.join(cmd))
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def evaluate_model(model_path, test_dir, output_dir='evaluation_results'):
    """
    Evaluate the trained model on test data using cellpose CLI.
    
    Args:
        model_path: Path to the trained model
        test_dir: Directory containing test images
        output_dir: Directory to save results
    """
    
    test_path = Path(test_dir) / 'test'
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Build cellpose evaluation command
    cmd = [
        'python', '-m', 'cellpose',
        '--dir', str(test_path),
        '--pretrained_model', str(model_path),
        '--chan', '0',
        '--chan2', '0',
        '--save_png',
        '--save_txt',
        '--verbose'
    ]
    
    print("Running cellpose evaluation with command:")
    print(' '.join(cmd))
    
    # Change to output directory to save results there
    original_cwd = os.getcwd()
    os.chdir(output_path)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    finally:
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(description='Train cellpose model using CLI')
    parser.add_argument('--data_dir', type=str, default='dataset/cellpose_format',
                        help='Directory containing train/, val/, test/ folders')
    parser.add_argument('--model_name', type=str, default='custom_cell_model',
                        help='Name for the saved model')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate existing model')
    parser.add_argument('--model_path', type=str,
                        help='Path to existing model for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        if not args.model_path:
            print("Error: --model_path required for evaluation")
            return
        
        print("Evaluating existing model...")
        success = evaluate_model(args.model_path, args.data_dir)
        if not success:
            print("Evaluation failed!")
            return
    
    else:
        # Train model
        print("Training model...")
        success = run_cellpose_training(
            data_dir=args.data_dir,
            model_name=args.model_name,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        if not success:
            print("Training failed!")
            return
        
        # Find the trained model
        model_files = list(Path('.').glob(f'models/{args.model_name}*'))
        if not model_files:
            model_files = list(Path('.').glob(f'{args.model_name}*'))
        
        if model_files:
            model_path = str(model_files[0])
            print(f"Found trained model: {model_path}")
            
            # Evaluate model
            print("Evaluating trained model...")
            evaluate_model(model_path, args.data_dir)
        else:
            print("Could not find trained model file")
    
    print("Done!")


if __name__ == "__main__":
    main()
