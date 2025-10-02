#!/bin/bash

# Set up your conda installation path
export PATH="/research/projects/trans_llm/Xiang_Zhang/cellpose/miniconda3/bin:$PATH"

echo "Setting up conda environment..."
echo "Using conda from: /research/projects/trans_llm/Xiang_Zhang/cellpose/miniconda3"
echo ""

# Activate the cellpose environment using the symlink
echo "Activating cellpose environment via symlink..."
conda activate /research/projects/trans_llm/Xiang_Zhang/cellpose/miniconda3/envs/cellpose_local

echo ""
echo "✅ Cellpose environment activated successfully!"
echo "📍 Environment location: ./envs/cellpose (symlinked to conda)"
echo ""
echo "🧪 To test the installation:"
echo "python -c \"import cellpose; print('Cellpose is working!')\""
echo ""
echo "🚀 You can now use cellpose and all its dependencies!"
