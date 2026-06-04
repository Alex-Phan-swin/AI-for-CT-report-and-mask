import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from scipy import ndimage

from model import UNet

def test_isolated_regions(checkpoint_path, image_path):
    """Test if model detects multiple isolated regions"""
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load and process image
    image = Image.open(image_path).convert("L").resize((256, 256))
    image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.15).float().squeeze().cpu().numpy()
    
    # Find connected components
    labeled, num_features = ndimage.label(mask)
    
    print(f"\n{'='*50}")
    print(f"ISOLATED REGION TEST")
    print(f"{'='*50}")
    print(f"Image: {Path(image_path).name}")
    print(f"Total connected components detected: {num_features}")
    
    if num_features == 0:
        print("\n❌ No regions detected. Model may need training or image may be tumour-free.")
    elif num_features == 1:
        print("\n⚠️ Only ONE region detected. For isolated region testing, you need:")
        print("   - An image with multiple disconnected tumour areas")
        print("   - OR train model on dataset with multifocal tumours")
    else:
        print(f"\n✓ SUCCESS: Found {num_features} isolated regions!")
        print("\nRegion details:")
        for i in range(1, num_features + 1):
            region_size = (labeled == i).sum()
            size_percent = (region_size / mask.size) * 100
            print(f"  Region {i}: {size_percent:.2f}% of image")
    
    # Also save the labeled regions image for inspection
    output_path = Path("outputs/isolated_regions_test.png")
    output_path.parent.mkdir(exist_ok=True)
    
    # Create colour image for regions
    colour_map = plt.cm.tab10(np.linspace(0, 1, num_features + 1))
    region_img = np.zeros((256, 256, 3))
    for i in range(1, num_features + 1):
        colour = colour_map[i][:3]
        region_img[labeled == i] = colour
    
    plt.imsave(output_path, region_img)
    print(f"\nSaved region visualization: {output_path}")
    
    return num_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/unet_brain_mri.pth")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    
    test_isolated_regions(args.checkpoint, args.image)