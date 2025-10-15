#!/usr/bin/env python3
"""
Script to generate and save a binary mask for a given image using GroundedSAM2.
Usage:
    python generate_mask.py \
        --image path/to/image.png \
        --ontology "robot arm" \
        --output mask.png
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

# add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, project_root)

from diffusion_policy.common.sam_util import generate_mask_from_image


def main():
    parser = argparse.ArgumentParser(description="Generate binary mask for specified objects in an image.")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='mask.png', help='Path to save binary mask (PNG)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    ontology = {'robot arm': 'robot arm'}  
    mask = generate_mask_from_image(args.image, ontology)

    # convert to 0-255 image
    mask_img = (mask * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    Image.fromarray(mask_img).save(args.output)
    print(f"Saved binary mask to {args.output}")


if __name__ == '__main__':
    main()
