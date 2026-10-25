from pathlib import Path
import random
import shutil

random.seed(42)

SOURCE = Path("data/split/train")
OUTPUT = Path("data/federated_balanced")

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing source directory: {SOURCE}")

# Re-create only this new experimental dataset.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for client in ("client1", "client2"):
    for student in range(1, 11):
        (OUTPUT / client / f"student{student}").mkdir(
            parents=True, exist_ok=True
        )

for student in range(1, 11):
    folder = SOURCE / f"student{student}"
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".pgm", ".bmp"
        }
    )

    random.shuffle(images)

    if len(images) < 2:
        raise ValueError(f"Not enough training images for {folder}")

    # Alternate images between clients, ensuring both get this class.
    for index, image in enumerate(images):
        client = "client1" if index % 2 == 0 else "client2"
        destination = OUTPUT / client / folder.name / image.name
        shutil.copy2(image, destination)

print("Balanced federated datasets created.")
for client in ("client1", "client2"):
    print(f"\n{client}:")
    total = 0
    for student in range(1, 11):
        folder = OUTPUT / client / f"student{student}"
        count = len(list(folder.iterdir()))
        total += count
        print(f"  student{student}: {count} images")
    print(f"Total images: {total}")

print("\nTest data remains unchanged in data/split/test.")
