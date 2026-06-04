import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from pathlib import Path

# Consistent colour mapping for report references
COLOUR_MAP = {
    "primary_tumour": (255, 0, 0),      # Red - main tumour mass
    "secondary_region": (0, 255, 0),    # Green - secondary/satellite regions  
    "suspicious_edge": (255, 255, 0),   # Yellow - suspicious boundaries
    "necrosis_core": (128, 0, 128),     # Purple - potential necrotic core
    "surrounding_oedema": (0, 255, 255) # Cyan - potential oedema region
}

def split_mask_into_regions(mask_array, min_region_size=50):
    """
    Split binary mask into connected components (isolated regions).
    Returns: list of (region_mask, region_colour, region_label)
    """
    # Label connected components
    labeled_mask, num_features = ndimage.label(mask_array)
    
    regions = []
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_mask == region_id).astype(np.uint8)
        region_size = region_mask.sum()
        
        if region_size >= min_region_size:
            # Assign colours based on region size/position
            if region_id == 1:
                colour = COLOUR_MAP["primary_tumour"]
                label = "primary tumour region"
            elif region_id == 2:
                colour = COLOUR_MAP["secondary_region"]
                label = "secondary tumour focus"
            else:
                colour = COLOUR_MAP["suspicious_edge"]
                label = f"suspicious region {region_id - 2}"
            
            regions.append({
                'mask': region_mask,
                'colour': colour,
                'label': label,
                'size_percent': (region_size / mask_array.size) * 100,
                'centre': ndimage.center_of_mass(region_mask)
            })
    
    return regions

def create_colour_coded_overlay(original_image, predicted_mask, output_path):
    """
    Create overlay with distinct colours for different regions.
    Also returns region data for report generation.
    """
    base = original_image.convert("RGB")
    base = base.resize((256, 256))
    base_array = np.array(base)
    
    # Split mask into regions
    mask_array = (predicted_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    regions = split_mask_into_regions(mask_array)
    
    # Create overlay with coloured regions
    overlay_array = base_array.copy()
    
    for region in regions:
        colour = np.array(region['colour'])
        region_mask = region['mask']
        
        # Apply semi-transparent colour to region
        alpha = 0.5
        for c in range(3):
            overlay_array[:, :, c][region_mask == 1] = (
                (1 - alpha) * overlay_array[:, :, c][region_mask == 1] + 
                alpha * colour[c]
            ).astype(np.uint8)
    
    # Draw outlines and labels
    overlay_img = Image.fromarray(overlay_array)
    draw = ImageDraw.Draw(overlay_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, region in enumerate(regions):
        cy, cx = region['centre']
        # Draw bounding box
        y_coords, x_coords = np.where(region['mask'] > 0)
        if len(x_coords) > 0:
            bbox = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
            draw.rectangle(bbox, outline=region['colour'], width=2)
            
            # Add label with colour name
            colour_name = [k for k, v in COLOUR_MAP.items() if v == region['colour']][0]
            label_text = f"{i+1}: {colour_name.replace('_', ' ')}"
            draw.text((bbox[0], bbox[1] - 15), label_text, fill=region['colour'], font=font)
    
    overlay_img.save(output_path)
    return regions

def create_legend_panel(output_path):
    """Create a standalone legend explaining colour coding"""
    legend_img = Image.new('RGB', (300, 200), 'white')
    draw = ImageDraw.Draw(legend_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    y = 10
    for colour_name, colour_rgb in COLOUR_MAP.items():
        draw.rectangle([10, y, 30, y+15], fill=colour_rgb, outline='black')
        draw.text((40, y), colour_name.replace('_', ' '), fill='black', font=font)
        y += 25
    
    legend_img.save(output_path)