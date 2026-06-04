import random
import shutil
from pathlib import Path


def sample_random_image(source_dir, demo_dir="demo_input"):
    source_dir = Path(source_dir)
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    images = list(source_dir.rglob("*.tif"))

    if not images:
        raise RuntimeError("No .tif images found in dataset")

    chosen = random.choice(images)

    dest = demo_dir / chosen.name
    shutil.copy2(chosen, dest)

    print(f"Selected sample:\n{chosen}\n→ {dest}")

    return dest