# Meshy AI Image to 3D Converter

This script converts images to 3D models using the Meshy AI API. It's currently configured to convert a Brawl Stars character cutout to a 3D model.

## Features

- Converts PNG/JPEG images to 3D models
- Supports multiple output formats (GLB, FBX, OBJ, USDZ)
- Automatic texture generation with customizable prompts
- Progress tracking during conversion
- Downloads all generated assets locally

## Setup

1. **Get a Meshy API Key**:
   - Sign up at [https://www.meshy.ai](https://www.meshy.ai)
   - Go to API settings: [https://www.meshy.ai/api](https://www.meshy.ai/api)
   - Generate an API key

2. **Configure API Key** (choose one method):
   - Set environment variable: `export MESHY_API_KEY=your_api_key_here`
   - Create a `.env` file with: `MESHY_API_KEY=your_api_key_here`
   - Enter it when prompted by the script
   - Use test mode (press Enter when prompted) - won't generate real results

3. **Install dependencies**:
   ```bash
   pip install requests
   ```

## Usage

Run the script:
```bash
python meshy_image_to_3d.py
```

The script will:
1. Load the Brawl Stars character image
2. Upload it to Meshy AI
3. Monitor the conversion progress
4. Download the generated 3D model and textures
5. Save everything to `meshy_3d_outputs/` directory

## Current Configuration

- **Input Image**: `data/brawl stars/25-06-25_at_02.38.41/analysis/universal_grounded_sam2/frame_detections/frame_0035/cutouts/rgba_cutout_000_player_0.31.png`
- **AI Model**: meshy-5 (latest)
- **Topology**: Quad-dominant mesh
- **Polycount**: 50,000 polygons
- **Texture Prompt**: "colorful cartoon game character with vibrant colors"

## Output Files

The script creates a timestamped directory with:
- `model.glb` - Universal 3D format (recommended)
- `model.fbx` - Autodesk format
- `model.obj` - Wavefront format
- `model.usdz` - Apple AR format
- `thumbnail.png` - Preview image
- `textures/` - Texture maps (base color, metallic, normal, roughness)
- `task_metadata.json` - Full API response data

## Customization

To convert a different image, modify these variables in the script:
- `IMAGE_PATH` - Path to your input image
- `payload["texture_prompt"]` - Text description for texturing
- `payload["target_polycount"]` - Polygon count (100-300,000)
- `payload["topology"]` - "quad" or "triangle"

## API Pricing

According to Meshy's documentation:
- Image to 3D (without texture): 5 credits
- Image to 3D (with texture): 15 credits

Check current pricing at: [https://www.meshy.ai/pricing](https://www.meshy.ai/pricing)

## Troubleshooting

- **"No API key found"**: Make sure you've set up your API key correctly
- **"Task failed"**: Check if the image is clear and shows a single object
- **"Error creating task: 401"**: Invalid API key
- **"Error creating task: 429"**: Rate limit exceeded, wait and try again

## Test Mode

The script supports Meshy's test mode API key (`msy_dummy_api_key_for_test_mode_12345678`). This allows you to test the integration without consuming credits, but it will return sample data instead of processing your actual image. 