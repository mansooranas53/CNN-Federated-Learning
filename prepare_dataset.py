from pathlib import Path
import random
import shutil

# Reproducible split
random.seed(42)

SOURCE_DIRS = [
    Path("data/client1"),
    Path("data/client2"),
]

TRAIN_DIR = Path("data/split/train")
TEST_DIR = Path("data/split/test")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}

# Create separate output directories
for directory in (TRAIN_DIR, TEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)

total_train = 0
total_test = 0

for source_dir in SOURCE_DIRS:
    if not source_dir.exists():
        raise FileNotFoundError(f"Folder not found: {source_dir}")

    for student_dir in sorted(source_dir.iterdir()):
        if not student_dir.is_dir():
            continue

        images = sorted(
            image for image in student_dir.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 2:
            print(f"Skipping {student_dir.name}: fewer than 2 images")
            continue

        random.shuffle(images)

        # Reserve at least one image for testing
        test_count = max(1, round(len(images) * 0.2))
        test_count = min(test_count, len(images) - 1)

        test_images = images[:test_count]
        train_images = images[test_count:]

        for split_dir, selected_images in (
            (TRAIN_DIR, train_images),
            (TEST_DIR, test_images),
        ):
            destination = split_dir / student_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in selected_images:
                shutil.copy2(
                    image_path,
                    destination / image_path.name
                )

        total_train += len(train_images)
        total_test += len(test_images)

        print(
            f"{student_dir.name}: "
            f"{len(train_images)} training, "
            f"{len(test_images)} testing"
        )

print("\nDataset split completed.")
print(f"Total training images: {total_train}")
print(f"Total testing images: {total_test}")
print(f"Training folder: {TRAIN_DIR}")
print(f"Testing folder: {TEST_DIR}")
print("\nOriginal client folders have not been modified.")
