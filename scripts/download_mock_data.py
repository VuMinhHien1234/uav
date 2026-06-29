"""
Generate mock UAV frames for demo — multi-terrain version.

5 terrains × 50 frames = 250 frames total.
Each terrain has a distinct visual style and a different level pattern
so the demo shows varied slow/medium/fast/skip events across environments.

Terrain patterns (designed to trigger different training scenarios):
  urban   — starts stable, escalates → many MEDIUM events (city navigation)
  forest  — high variation throughout → mix of FAST + MEDIUM
  desert  — long stable stretches then sudden drift → dramatic SLOW events
  coastal — alternating stable/change → balanced FAST/MEDIUM mix
  mountain — extreme variation, frequent SLOW → most retrains

Run from project root:
    python3 scripts/download_mock_data.py
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR     = "mock-data"
FRAMES_PER_ENV = 50
IMG_SIZE       = (224, 224)

# ── Terrain definitions ───────────────────────────────────────────────────────

ENVS = {
    "urban":    {"base": (70,  100, 160), "name": "Urban",    "accent": (200, 200, 210)},
    "forest":   {"base": (34,  110,  34), "name": "Forest",   "accent": (180, 220, 100)},
    "desert":   {"base": (210, 170,  90), "name": "Desert",   "accent": (240, 200, 130)},
    "coastal":  {"base": (40,  140, 200), "name": "Coastal",  "accent": (220, 240, 255)},
    "mountain": {"base": (100,  90,  80), "name": "Mountain", "accent": (200, 195, 190)},
}

# ── Level patterns per terrain ────────────────────────────────────────────────
# Each terrain triggers a different sequence of NL levels for demo variety.
# "skip"   = very similar to previous (delta < NL_FAST_DELTA)
# "fast"   = moderate delta
# "medium" = large delta (scene change)
# "slow"   = accumulated drift threshold

def _urban_pattern():
    """Stable start, escalating changes — many medium events."""
    p = {}
    # Frames 0-9: stable warm-up
    for i in range(0, 10):
        p[i] = "skip" if i % 2 == 0 else "fast"
    # Frames 10-19: medium changes
    for i in range(10, 20):
        p[i] = "medium" if i % 3 == 0 else "fast"
    # Frame 20: slow accumulation
    p[20] = "slow"
    # Frames 21-34: unstable urban traffic
    for i in range(21, 35):
        p[i] = "medium" if i % 4 == 0 else "fast"
    # Frame 35: slow again
    p[35] = "slow"
    # Frames 36-49: wind-down, mostly fast
    for i in range(36, 50):
        p[i] = "skip" if i % 3 == 0 else "fast"
    return p


def _forest_pattern():
    """High variation — frequent fast + medium from canopy/lighting changes."""
    p = {}
    for i in range(50):
        if i % 15 == 0:
            p[i] = "slow"
        elif i % 5 == 0:
            p[i] = "medium"
        elif i % 2 == 0:
            p[i] = "fast"
        else:
            p[i] = "skip"
    return p


def _desert_pattern():
    """Long stable stretches, then sudden dramatic drift → big SLOW events."""
    p = {}
    # Very stable: 0-14
    for i in range(0, 15):
        p[i] = "skip"
    # Sudden storm/drift: 15-19
    for i in range(15, 20):
        p[i] = "medium" if i % 2 == 0 else "fast"
    p[20] = "slow"
    # Stable again: 21-34
    for i in range(21, 35):
        p[i] = "skip" if i % 2 == 0 else "fast"
    # Second dramatic event: 35-40
    for i in range(35, 41):
        p[i] = "medium"
    p[41] = "slow"
    # Recovery: 42-49
    for i in range(42, 50):
        p[i] = "skip" if i % 3 == 0 else "fast"
    return p


def _coastal_pattern():
    """Alternating stable/change cycles — balanced mix."""
    p = {}
    for i in range(50):
        cycle = i % 10
        if cycle < 4:
            p[i] = "skip"
        elif cycle < 7:
            p[i] = "fast"
        elif cycle == 7:
            p[i] = "medium"
        elif cycle == 8:
            p[i] = "fast"
        else:
            p[i] = "slow" if i > 20 else "medium"
    return p


def _mountain_pattern():
    """Extreme variation, most retrains — hardest terrain."""
    p = {}
    for i in range(50):
        if i % 8 == 0:
            p[i] = "slow"
        elif i % 3 == 0:
            p[i] = "medium"
        elif i % 2 == 0:
            p[i] = "fast"
        else:
            p[i] = "fast"
    return p


PATTERNS = {
    "urban":    _urban_pattern(),
    "forest":   _forest_pattern(),
    "desert":   _desert_pattern(),
    "coastal":  _coastal_pattern(),
    "mountain": _mountain_pattern(),
}

# ── Terrain-specific visual generators ───────────────────────────────────────

def _draw_urban(draw, size, color):
    """Buildings, windows, roads."""
    w, h = size
    # Road
    draw.rectangle([w//2 - 20, 0, w//2 + 20, h], fill=(60, 60, 60))
    # Buildings
    for _ in range(random.randint(3, 6)):
        bw = random.randint(20, 50)
        bh = random.randint(40, 120)
        bx = random.randint(0, w - bw)
        draw.rectangle([bx, h - bh, bx + bw, h], fill=color)
        # Windows
        for wy in range(h - bh + 5, h - 10, 15):
            for wx in range(bx + 5, bx + bw - 5, 12):
                draw.rectangle([wx, wy, wx + 8, wy + 10],
                                fill=(255, 255, 180) if random.random() > 0.3 else (40, 40, 60))


def _draw_forest(draw, size, color):
    """Trees, canopy, ground."""
    w, h = size
    # Ground
    draw.rectangle([0, h * 3 // 4, w, h], fill=(60, 80, 30))
    # Trees
    for _ in range(random.randint(4, 8)):
        tx = random.randint(0, w)
        ty = random.randint(h // 4, h * 3 // 4)
        radius = random.randint(15, 40)
        trunk_h = random.randint(10, 30)
        draw.rectangle([tx - 4, ty, tx + 4, ty + trunk_h], fill=(80, 50, 20))
        draw.ellipse([tx - radius, ty - radius, tx + radius, ty], fill=color)


def _draw_desert(draw, size, color):
    """Sand dunes, sparse rocks."""
    w, h = size
    # Dunes
    for i in range(3):
        cx = random.randint(20, w - 20)
        cy = h - random.randint(20, 80)
        draw.ellipse([cx - 60, cy - 20, cx + 60, cy + 20], fill=color)
    # Rocks
    for _ in range(random.randint(2, 5)):
        rx = random.randint(0, w)
        ry = random.randint(h // 2, h)
        rs = random.randint(5, 20)
        draw.ellipse([rx, ry, rx + rs, ry + rs * 0.6],
                     fill=(140, 120, 90))


def _draw_coastal(draw, size, color):
    """Water, waves, horizon."""
    w, h = size
    # Sky gradient (already in background)
    # Water
    draw.rectangle([0, h // 2, w, h], fill=(30, 100, 180))
    # Waves
    for i in range(0, h, 20):
        if i > h // 2:
            offset = random.randint(-5, 5)
            draw.arc([offset, i, w + offset, i + 15],
                     start=0, end=180, fill=(200, 230, 255), width=2)
    # Beach
    draw.rectangle([0, h * 2 // 3, w, h * 3 // 4], fill=(220, 200, 150))


def _draw_mountain(draw, size, color):
    """Peaks, ridges, snow caps."""
    w, h = size
    # Mountain peaks
    for _ in range(random.randint(2, 4)):
        px = random.randint(0, w)
        ph = random.randint(h // 3, h * 2 // 3)
        draw.polygon([(px - 40, h), (px, ph), (px + 40, h)], fill=color)
        # Snow cap
        draw.polygon([(px - 12, ph + 20), (px, ph), (px + 12, ph + 20)],
                     fill=(240, 240, 255))
    # Ridge line
    points = [(0, h * 2 // 3)]
    for x in range(0, w, 20):
        points.append((x, h * 2 // 3 + random.randint(-30, 30)))
    points.append((w, h * 2 // 3))
    points.append((w, h))
    points.append((0, h))
    draw.polygon(points, fill=(80, 75, 70))


DRAWERS = {
    "urban":    _draw_urban,
    "forest":   _draw_forest,
    "desert":   _draw_desert,
    "coastal":  _draw_coastal,
    "mountain": _draw_mountain,
}

# ── Frame generator ───────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "fast":   (100, 255, 100),
    "medium": (255, 220,  50),
    "slow":   (255,  80,  80),
    "skip":   (150, 150, 150),
}


def make_frame(env_key: str, frame_idx: int, cfg: dict,
               frame_type: str, prev_color: tuple) -> tuple:
    r, g, b = cfg["base"]

    # Color variation by frame type
    if frame_type == "skip":
        r2, g2, b2 = prev_color
        noise_range = 10
        n_extra = 0
    elif frame_type == "fast":
        shift = 35
        r2 = int(np.clip(r + random.randint(-shift, shift), 0, 255))
        g2 = int(np.clip(g + random.randint(-shift, shift), 0, 255))
        b2 = int(np.clip(b + random.randint(-shift, shift), 0, 255))
        noise_range = 40
        n_extra = random.randint(1, 3)
    elif frame_type == "medium":
        shift = 80
        r2 = int(np.clip(r + random.randint(-shift, shift), 0, 255))
        g2 = int(np.clip(g + random.randint(-shift, shift), 0, 255))
        b2 = int(np.clip(b + random.randint(-shift, shift), 0, 255))
        noise_range = 70
        n_extra = random.randint(3, 6)
    else:  # slow
        shift = 120
        r2 = int(np.clip(r + random.randint(-shift, shift), 0, 255))
        g2 = int(np.clip(g + random.randint(-shift, shift), 0, 255))
        b2 = int(np.clip(b + random.randint(-shift, shift), 0, 255))
        noise_range = 90
        n_extra = random.randint(6, 10)

    # Base image
    noise = np.random.randint(-noise_range, noise_range,
                               (IMG_SIZE[1], IMG_SIZE[0], 3))
    base  = np.full((IMG_SIZE[1], IMG_SIZE[0], 3), [r2, g2, b2])
    arr   = np.clip(base + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)
    draw  = ImageDraw.Draw(img)

    # Terrain-specific shapes
    DRAWERS[env_key](draw, IMG_SIZE, cfg["accent"])

    # Extra random shapes for higher-level frames
    for _ in range(n_extra):
        x1 = random.randint(0, 160)
        y1 = random.randint(50, 160)
        x2 = x1 + random.randint(15, 60)
        y2 = y1 + random.randint(15, 60)
        c  = tuple(random.randint(0, 255) for _ in range(3))
        if random.random() > 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=c)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=c)

    # HUD overlay
    draw.rectangle([0, 0, IMG_SIZE[0], 44], fill=(0, 0, 0, 180))
    draw.text((6,  4), f"{cfg['name']} | Frame {frame_idx:03d}/{FRAMES_PER_ENV}",
              fill=(255, 255, 255))
    draw.text((6, 24), f"[{frame_type.upper()}]",
              fill=LEVEL_COLORS.get(frame_type, (255, 255, 255)))

    return img, (r2, g2, b2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Generating mock UAV frames → {OUTPUT_DIR}/")
    print(f"  {len(ENVS)} terrains × {FRAMES_PER_ENV} frames = {len(ENVS) * FRAMES_PER_ENV} total\n")

    for env_key, cfg in ENVS.items():
        env_dir = os.path.join(OUTPUT_DIR, env_key)
        os.makedirs(env_dir, exist_ok=True)

        pattern    = PATTERNS[env_key]
        prev_color = cfg["base"]
        counts     = {"skip": 0, "fast": 0, "medium": 0, "slow": 0}

        for i in range(FRAMES_PER_ENV):
            ftype = pattern.get(i, "fast")
            counts[ftype] += 1
            img, prev_color = make_frame(env_key, i, cfg, ftype, prev_color)
            img.save(os.path.join(env_dir, f"frame_{i:03d}.jpg"), quality=85)

        print(
            f"  ✓ {env_key:10s} ({cfg['name']:8s})  "
            f"skip={counts['skip']:2d}  fast={counts['fast']:2d}  "
            f"medium={counts['medium']:2d}  slow={counts['slow']:2d}"
        )

    print(f"\nDone — run: python3 -m simulator.flight_agent")


if __name__ == "__main__":
    main()
