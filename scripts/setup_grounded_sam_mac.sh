#!/bin/bash
# Setup script for Grounded-SAM-2 on macOS
# This script clones the repository, installs dependencies, and downloads required models

set -e  # Exit on error

echo "🚀 Grounded-SAM-2 Setup Script for macOS"
echo "========================================"

# Check if we're in the project root
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
    echo "❌ Error: This script must be run from the project root directory"
    echo "Please cd to the roam-game-analysis directory and try again"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.11 or 3.12 first"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Found Python $PYTHON_VERSION"

# Check if Grounded-SAM-2 already exists
if [ -d "Grounded-SAM-2" ]; then
    echo "⚠️  Grounded-SAM-2 directory already exists"
    read -p "Do you want to remove it and start fresh? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf Grounded-SAM-2
        echo "✅ Removed existing Grounded-SAM-2 directory"
    else
        echo "ℹ️  Keeping existing directory, skipping clone step"
    fi
fi

# Clone the repository if it doesn't exist
if [ ! -d "Grounded-SAM-2" ]; then
    echo "📦 Cloning Grounded-SAM-2 repository..."
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
    echo "✅ Repository cloned successfully"
fi

cd Grounded-SAM-2

# Create necessary directories
echo "📁 Creating model directories..."
mkdir -p checkpoints gdino_checkpoints

# Download SAM 2 models
echo "⬇️  Downloading SAM 2 models..."
cd checkpoints

if [ ! -f "sam2_hiera_large.pt" ]; then
    echo "  📥 Downloading sam2_hiera_large.pt (896MB)..."
    curl -L -o sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    echo "  ✅ sam2_hiera_large.pt downloaded"
else
    echo "  ✅ sam2_hiera_large.pt already exists"
fi

if [ ! -f "sam2_hiera_base_plus.pt" ]; then
    echo "  📥 Downloading sam2_hiera_base_plus.pt (320MB)..."
    curl -L -o sam2_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
    echo "  ✅ sam2_hiera_base_plus.pt downloaded"
else
    echo "  ✅ sam2_hiera_base_plus.pt already exists"
fi

cd ..

# Download Grounding DINO models
echo "⬇️  Downloading Grounding DINO models..."
cd gdino_checkpoints

if [ ! -f "groundingdino_swinb_cogcoor.pth" ]; then
    echo "  📥 Downloading groundingdino_swinb_cogcoor.pth (662MB)..."
    curl -L -o groundingdino_swinb_cogcoor.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
    echo "  ✅ groundingdino_swinb_cogcoor.pth downloaded"
else
    echo "  ✅ groundingdino_swinb_cogcoor.pth already exists"
fi

cd ../..

# Verify MPS support (Apple Silicon)
echo ""
echo "🖥️  Checking Apple Silicon MPS support..."
python3 -c "import torch; mps_available = torch.backends.mps.is_available(); print(f'{'✅' if mps_available else '⚠️ '} MPS (Metal Performance Shaders) available: {mps_available}')"

# Final setup instructions
echo ""
echo "✨ Grounded-SAM-2 setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Run 'uv sync' to install all Python dependencies"
echo "2. Test the installation with:"
echo "   uv run python -m src.analysis.video_behavior_analysis.run_improved_analysis"
echo ""
echo "💡 Tips for Mac users:"
echo "- MPS acceleration will be used automatically on Apple Silicon"
echo "- If you encounter memory issues, use smaller models or reduce batch size"
echo "- Performance: M1/M2 ~0.5-1 FPS, M2/M3 Pro/Max ~1-2 FPS"
echo ""
echo "🔧 For troubleshooting, see:"
echo "   src/analysis/video_behavior_analysis/SETUP_GROUNDED_SAM.md"