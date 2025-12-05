from PIL import Image, ImageDraw, ImageFont
import pycocotools.mask as mask_utils
import numpy as np
import random


def visualize_masks_on_image(image, masks):
    """Visualize RLE masks on image with region labels."""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Try to load font, fallback to default if not available
    try:
        font = ImageFont.truetype("data/dejavu/DejaVuSans-Bold.ttf", 25)
    except ValueError:
        font = ImageFont.load_default()

    text_infos = []

    # Process each mask
    for i, mask_rle in enumerate(masks):
        # Decode RLE mask
        mask = mask_utils.decode(mask_rle)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        # Random color for each region
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            128,
        )
        colored_mask = Image.new("RGBA", image.size, color)
        overlay.paste(colored_mask, (0, 0), mask_image)

        # Calculate center position for text
        mask_indices = np.argwhere(mask)
        if mask_indices.size > 0:
            min_y, min_x = mask_indices.min(axis=0)
            max_y, max_x = mask_indices.max(axis=0)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            text = f"Region {i}"
            draw = ImageDraw.Draw(overlay)
            text_size = draw.textbbox((0, 0), text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            text_position = (center_x - text_width // 2, center_y - text_height // 2)
            text_infos.append((text, text_position))

    # Draw all text labels
    draw = ImageDraw.Draw(overlay)
    for text, text_position in text_infos:
        draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)

    # Blend image with overlay
    blended_image = Image.alpha_composite(image, overlay)
    return blended_image.convert("RGB")
