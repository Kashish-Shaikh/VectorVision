from pathlib import Path
import json

from PIL import Image
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _collect_pairs(root: Path):
    """
    FloodNet Track 1 structure:

    Train/
      Labeled/
        Flooded/
          image/*.jpg
          mask/*_lab.png
        Non-Flooded/
          image/*.jpg
          mask/*_lab.png

    Images and masks are paired by numeric filename stem.
    """

    pairs = []

    for category in ["Flooded", "Non-Flooded"]:
        image_dir = root / "Train" / "Labeled" / category / "image"
        mask_dir = root / "Train" / "Labeled" / category / "mask"

        if not image_dir.exists():
            continue

        if not mask_dir.exists():
            continue

        images = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }

        masks = {
            p.stem.replace("_lab", ""): p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }

        common = sorted(set(images) & set(masks))

        for stem in common:
            pairs.append(
                {
                    "image": images[stem],
                    "mask": masks[stem],
                    "category": category,
                }
            )

    return pairs


def _find_test_images(root: Path):
    test_dir = root / "Test" / "image"

    if not test_dir.exists():
        return []

    return sorted(
        p
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _find_validation_images(root: Path):
    val_dir = root / "Validation" / "image"

    if not val_dir.exists():
        return []

    return sorted(
        p
        for p in val_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def summarise(root):
    """Inspect the FloodNet Track 1 dataset."""

    root = Path(root)

    print("FloodNet Track 1 dataset")
    print("=" * 72)
    print("root:", root)

    if not root.exists():
        print("ERROR: dataset root does not exist")
        return

    pairs = _collect_pairs(root)

    flooded = [p for p in pairs if p["category"] == "Flooded"]
    non_flooded = [p for p in pairs if p["category"] == "Non-Flooded"]

    test_images = _find_test_images(root)
    val_images = _find_validation_images(root)

    print()
    print("Labeled training data:")
    print(f"  Flooded     : {len(flooded)} image/mask pairs")
    print(f"  Non-Flooded : {len(non_flooded)} image/mask pairs")
    print(f"  TOTAL       : {len(pairs)} image/mask pairs")

    print()
    print("Other splits:")
    print(f"  Validation images : {len(val_images)}")
    print(f"  Test images       : {len(test_images)}")

    if pairs:
        print()
        print("Example pairs:")
        for item in pairs[:5]:
            print(
                f"  {item['image'].name} <--> "
                f"{item['mask'].name} "
                f"({item['category']})"
            )

    print("=" * 72)


def _load_mask(path: Path):
    """
    Load a FloodNet label mask.

    The raw mask contains class IDs. The training target is converted
    to binary water / non-water according to water_classes in config.
    """

    return np.asarray(Image.open(path))


def _load_image(path: Path, image_size: int):
    """
    Load RGB image and resize it to the requested square size.
    """

    image = Image.open(path).convert("RGB")
    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _resize_mask(mask, image_size: int):
    """
    Resize segmentation mask using nearest-neighbour interpolation.
    """

    img = Image.fromarray(mask.astype(np.uint8))
    img = img.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST,
    )

    return np.asarray(img)


def build_cache(cfg, root, cache_dir, force=False):
    """
    Build a compact NPZ cache from the labeled FloodNet training data.

    The cache contains:
        images : RGB float32 [N,H,W,3]
        masks  : binary uint8 [N,H,W]
        labels : frame-level water presence
        names  : source filenames
        groups : Flooded / Non-Flooded
    """

    root = Path(root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "floodnet_train.npz"
    meta_file = cache_dir / "metadata.json"

    if cache_file.exists() and not force:
        print(f"Using existing cache: {cache_file}")
        return cache_file

    image_size = int(cfg["drone_unet"]["image_size"])

    # Support the requested config key while retaining compatibility
    # with the earlier configuration.
    water_classes = cfg["drone_unet"].get(
        "water_class_ids",
        cfg["drone_unet"].get("water_classes", []),
    )

    pairs = _collect_pairs(root)

    if not pairs:
        raise RuntimeError(
            "No FloodNet image/mask pairs found. "
            "Check the FloodNet root and folder structure."
        )

    print("=" * 72)
    print("BUILDING FLOODNET CACHE")
    print("=" * 72)
    print(f"source pairs : {len(pairs)}")
    print(f"image size   : {image_size}x{image_size}")
    print(f"water classes: {water_classes}")

    images = []
    masks = []
    labels = []
    names = []
    categories = []

    for i, item in enumerate(pairs, start=1):

        image = _load_image(item["image"], image_size)

        raw_mask = _load_mask(item["mask"])
        raw_mask = _resize_mask(raw_mask, image_size)

        binary = np.isin(raw_mask, water_classes).astype(np.uint8)

        frame_has_water = int(binary.sum() > 0)

        images.append(image)
        masks.append(binary)
        labels.append(frame_has_water)
        names.append(item["image"].name)
        categories.append(item["category"])

        if i % 50 == 0 or i == len(pairs):
            print(f"processed {i}/{len(pairs)}")

    images = np.stack(images).astype(np.float32)
    masks = np.stack(masks).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)

    np.savez_compressed(
        cache_file,
        images=images,
        masks=masks,
        labels=labels,
        names=np.asarray(names),
        categories=np.asarray(categories),
    )

    metadata = {
        "source": str(root),
        "num_pairs": len(pairs),
        "image_size": image_size,
        "water_classes": list(water_classes),
        "water_frames": int(labels.sum()),
        "non_water_frames": int((labels == 0).sum()),
    }

    meta_file.write_text(json.dumps(metadata, indent=2))

    print()
    print("CACHE COMPLETE")
    print(f"cache : {cache_file}")
    print(f"meta  : {meta_file}")
    print(f"images: {len(images)}")
    print(f"water frames: {int(labels.sum())}")
    print(f"non-water frames: {int((labels == 0).sum())}")
    print("=" * 72)

    return cache_file