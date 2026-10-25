from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.models.model import GhostFaceNetV2

# Configuration
NUM_CLASSES = 10
TEST_DIR = Path("data/split/test")
CHECKPOINT_PATH = Path(
    "experiments/exp_01_baseline/checkpoints/final_model.pth"
)
BATCH_SIZE = 8

LABEL_MAP = {
    f"student{i}": i - 1
    for i in range(1, NUM_CLASSES + 1)
}
CLASS_NAMES = [
    f"student{i}"
    for i in range(1, NUM_CLASSES + 1)
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}


class FaceTestDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = []
        self.labels = []

        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Test directory not found: {data_dir}"
            )

        for person_dir in sorted(data_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person = person_dir.name

            if person not in LABEL_MAP:
                continue

            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.image_paths.append(image_path)
                    self.labels.append(LABEL_MAP[person])

        if not self.image_paths:
            raise ValueError("No test images found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label


def main():
    device = torch.device("cpu")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("Checkpoint class count is incorrect.")

    model = GhostFaceNetV2(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = FaceTestDataset(TEST_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    print("=" * 55)
    print("FEDERATED LEARNING MODEL - TEST EVALUATION")
    print("=" * 55)
    print(f"Test images: {len(dataset)}")
    print(f"Number of classes: {NUM_CLASSES}")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = correct / total if total else 0.0

    print(f"\nCorrect predictions: {correct}/{total}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    print("CONFUSION MATRIX")
    print("Rows = actual students; columns = predicted students")
    print("Student labels:", CLASS_NAMES)
    print(matrix)

    # Save evaluation results
    results_dir = Path("experiments/exp_01_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "evaluation_results.txt"

    with results_path.open("w", encoding="utf-8") as f:
        f.write("FEDERATED LEARNING MODEL - TEST EVALUATION\n")
        f.write(f"Test images: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Test accuracy: {accuracy * 100:.2f}%\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write(
            classification_report(
                all_labels,
                all_predictions,
                labels=list(range(NUM_CLASSES)),
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

        f.write("\nCONFUSION MATRIX\n")
        f.write(str(matrix))
        f.write("\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
