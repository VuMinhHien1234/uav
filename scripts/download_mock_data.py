"""
Generate mock UAV frames for demo.
Creates 3 environments × 30 frames = 90 colour images (no internet required).

env1 (urban)  → blue tones  (buildings)
env2 (forest) → green tones (trees)
env3 (desert) → sandy tones (sand)

Within each env: small noise added per frame for visual variation.

Run from project root:
    python3 scripts/download_mock_data.py
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR     = "mock-data"
FRAMES_PER_ENV = 30
IMG_SIZE       = (224, 224)

ENVS = {
    "env1": {"base": (70, 130, 180),  "name": "Urban"},
    "env2": {"base": (34, 139, 34),   "name": "Forest"},
    "env3": {"base": (205, 170, 100), "name": "Desert"},
}


def make_frame(env_key: str, frame_idx: int, base_color: tuple) -> Image.Image:
    r, g, b = base_color
    noise     = np.random.randint(-20, 20, (IMG_SIZE[1], IMG_SIZE[0], 3))
    img_array = np.clip(
        np.full((IMG_SIZE[1], IMG_SIZE[0], 3), [r, g, b]) + noise, 0, 255
    ).astype(np.uint8)

    img  = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    draw.rectangle([5, 5, 219, 45], fill=(0, 0, 0, 180))
    draw.text((10, 10), ENVS[env_key]["name"],    fill=(255, 255, 255))
    draw.text((10, 25), f"Frame {frame_idx:03d}", fill=(200, 200, 200))

    if random.random() > 0.6:
        ox = random.randint(30, 180)
        oy = random.randint(60, 180)
        draw.ellipse([ox, oy, ox + 25, oy + 25], fill=(255, 80, 80))

    return img


def main():
    print(f"Generating mock UAV frames in {OUTPUT_DIR}/...")

    for env_key, cfg in ENVS.items():
        env_dir = os.path.join(OUTPUT_DIR, env_key)
        os.makedirs(env_dir, exist_ok=True)

        for i in range(FRAMES_PER_ENV):
            img  = make_frame(env_key, i, cfg["base"])
            path = os.path.join(env_dir, f"frame_{i:03d}.jpg")
            img.save(path, quality=85)

        print(f"  ✓ {env_key} ({cfg['name']}): {FRAMES_PER_ENV} frames → {env_dir}/")

    total = len(ENVS) * FRAMES_PER_ENV
    print(f"\nDone: {total} frames in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
