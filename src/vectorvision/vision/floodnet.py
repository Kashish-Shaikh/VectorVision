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

        if not image_dir.exists() or not mask_dir.exists():
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
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _find_validation_images(root: Path):
    val_dir = root / "Validation" / "image"

    if not val_dir.exists():
        return []

    return sorted(
        p for p in val_dir.iterdir()
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
    """Load the raw FloodNet class-ID mask."""

    return np.asarray(Image.open(path))


def _load_image(path: Path, image_size: int):
    """
    Load RGB image and resize it.

    IMPORTANT:
    Keep pixels as uint8 [0,255].

    train_drone.py performs the /255 normalization.
    """

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(image, dtype=np.uint8)


def _resize_mask(mask, image_size: int):
    """Resize segmentation mask using nearest-neighbour interpolation."""

    img = Image.fromarray(mask.astype(np.uint8))

    img = img.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST,
    )

    return np.asarray(img)


def _split_indices(pairs, seed=42):
    """
    Deterministic 70/15/15 split.

    The split is performed separately for Flooded and Non-Flooded
    categories so both categories are represented in train/val/test.
    """

    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    for category in ["Flooded", "Non-Flooded"]:

        idx = np.array(
            [i for i, p in enumerate(pairs)
             if p["category"] == category],
            dtype=int,
        )

        rng.shuffle(idx)

        n = len(idx)

        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        # Make sure every split receives at least one sample.
        n_train = max(1, n_train)
        n_val = max(1, n_val)

        train_part = idx[:n_train]
        val_part = idx[n_train:n_train + n_val]
        test_part = idx[n_train + n_val:]

        train_idx.extend(train_part.tolist())
        val_idx.extend(val_part.tolist())
        test_idx.extend(test_part.tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


def _build_arrays(pairs, indices, image_size, water_classes, split_name):
    """Load and convert a selected split."""

    images = []
    masks = []
    labels = []
    names = []
    categories = []

    print()
    print(f"Building {split_name} split: {len(indices)} images")

    for count, idx in enumerate(indices, start=1):

        item = pairs[int(idx)]

        image = _load_image(
            item["image"],
            image_size,
        )

        raw_mask = _load_mask(item["mask"])

        raw_mask = _resize_mask(
            raw_mask,
            image_size,
        )

        binary = np.isin(
            raw_mask,
            water_classes,
        ).astype(np.uint8)

        frame_has_water = int(binary.sum() > 0)

        images.append(image)
        masks.append(binary)
        labels.append(frame_has_water)
        names.append(item["image"].name)
        categories.append(item["category"])

        if count % 50 == 0 or count == len(indices):
            print(
                f"  processed {count}/{len(indices)}"
            )

    return (
        np.stack(images).astype(np.uint8),
        np.stack(masks).astype(np.uint8),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(names),
        np.asarray(categories),
    )


def build_cache(cfg, root, cache_dir, force=False):
    """
    Build train/validation/internal-test caches from the 398
    labeled FloodNet Track 1 pairs.

    IMPORTANT:
    The official FloodNet Validation/Test folders contain images
    without public masks in this dataset. Therefore they cannot be
    used for segmentation metrics here.

    Instead, the 398 labeled pairs are split internally:

        70% train
        15% validation
        15% internal test
    """

    root = Path(root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_file = cache_dir / "floodnet_meta.json"

    required = [
        cache_dir / "X_train.npy",
        cache_dir / "Y_train.npy",
        cache_dir / "X_val.npy",
        cache_dir / "Y_val.npy",
        cache_dir / "X_test.npy",
        cache_dir / "Y_test.npy",
    ]

    if all(p.exists() for p in required) and meta_file.exists() and not force:
        print("=" * 72)
        print("USING EXISTING FLOODNET CACHE")
        print("=" * 72)
        print(f"cache: {cache_dir}")
        return cache_dir

    image_size = int(
        cfg["drone_unet"]["image_size"]
    )

    water_classes = cfg["drone_unet"].get(
        "water_class_ids",
        cfg["drone_unet"].get("water_classes", []),
    )

    water_classes = [int(x) for x in water_classes]

    pairs = _collect_pairs(root)

    if not pairs:
        raise RuntimeError(
            "No FloodNet image/mask pairs found. "
            "Check the FloodNet root and folder structure."
        )

    seed = int(
        cfg.get("project", {}).get("seed", 42)
    )

    print("=" * 72)
    print("BUILDING FLOODNET CACHE")
    print("=" * 72)
    print(f"source pairs : {len(pairs)}")
    print(f"image size   : {image_size}x{image_size}")
    print(f"water classes: {water_classes}")
    print("split        : 70% train / 15% validation / 15% test")
    print("test         : INTERNAL HOLD-OUT FROM LABELED DATA")
    print("=" * 72)

    train_idx, val_idx, test_idx = _split_indices(
        pairs,
        seed=seed,
    )

    print()
    print("SPLIT SIZES")
    print(f"  train : {len(train_idx)}")
    print(f"  val   : {len(val_idx)}")
    print(f"  test  : {len(test_idx)}")
    print(f"  total : {len(train_idx) + len(val_idx) + len(test_idx)}")

    X_train, Y_train, L_train, N_train, C_train = _build_arrays(
        pairs,
        train_idx,
        image_size,
        water_classes,
        "TRAIN",
    )

    X_val, Y_val, L_val, N_val, C_val = _build_arrays(
        pairs,
        val_idx,
        image_size,
        water_classes,
        "VALIDATION",
    )

    X_test, Y_test, L_test, N_test, C_test = _build_arrays(
        pairs,
        test_idx,
        image_size,
        water_classes,
        "INTERNAL TEST",
    )

    np.save(cache_dir / "X_train.npy", X_train)
    np.save(cache_dir / "Y_train.npy", Y_train)

    np.save(cache_dir / "X_val.npy", X_val)
    np.save(cache_dir / "Y_val.npy", Y_val)

    np.save(cache_dir / "X_test.npy", X_test)
    np.save(cache_dir / "Y_test.npy", Y_test)

    np.save(cache_dir / "labels_train.npy", L_train)
    np.save(cache_dir / "labels_val.npy", L_val)
    np.save(cache_dir / "labels_test.npy", L_test)

    np.save(cache_dir / "names_train.npy", N_train)
    np.save(cache_dir / "names_val.npy", N_val)
    np.save(cache_dir / "names_test.npy", N_test)

    metadata = {
        "source": str(root),
        "num_pairs": len(pairs),
        "image_size": image_size,
        "water_classes": water_classes,

        "split": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "internal_test": int(len(test_idx)),
            "seed": seed,
            "fractions": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
        },

        "train_water_frames": int(L_train.sum()),
        "train_non_water_frames": int(
            (L_train == 0).sum()
        ),

        "val_water_frames": int(L_val.sum()),
        "val_non_water_frames": int(
            (L_val == 0).sum()
        ),

        "test_water_frames": int(L_test.sum()),
        "test_non_water_frames": int(
            (L_test == 0).sum()
        ),

        "official_validation_images": len(
            _find_validation_images(root)
        ),

        "official_test_images": len(
            _find_test_images(root)
        ),

        "test_type": "internal_holdout_from_398_labeled_pairs",
    }

    meta_file.write_text(
        json.dumps(metadata, indent=2)
    )

    print()
    print("=" * 72)
    print("FLOODNET CACHE COMPLETE")
    print("=" * 72)
    print(f"cache directory : {cache_dir}")
    print(f"train           : {len(X_train)}")
    print(f"validation      : {len(X_val)}")
    print(f"internal test   : {len(X_test)}")
    print()
    print(
        f"train water frames: "
        f"{int(L_train.sum())}/{len(L_train)}"
    )
    print(
        f"val water frames: "
        f"{int(L_val.sum())}/{len(L_val)}"
    )
    print(
        f"test water frames: "
        f"{int(L_test.sum())}/{len(L_test)}"
    )
    print("=" * 72)

    return cache_dir