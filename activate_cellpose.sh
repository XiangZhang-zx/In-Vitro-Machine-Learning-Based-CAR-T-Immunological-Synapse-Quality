#!/bin/bash

# Set up conda path
export PATH="/research/projects/trans_llm/Xiang_Zhang/cellpose/miniconda3/bin:$PATH"

echo "Activating cellpose environment..."
echo "Method 1: Using local path"
echo "conda activate ./envs/cellpose"
echo ""
echo "Method 2: Using conda with symlink"
echo "conda activate /research/projects/trans_llm/Xiang_Zhang/cellpose/miniconda3/envs/cellpose_local"
echo ""
echo "Method 3: Direct activation (recommended)"
conda activate ./envs/cellpose

echo "Cellpose environment activated!"
echo "You can now use cellpose and all its dependencies."
echo ""
echo "To test the installation, try:"
echo "python -c \"import cellpose; print('Cellpose is working!')\""
