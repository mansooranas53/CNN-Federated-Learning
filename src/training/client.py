import sys
from pathlib import Path

import cv2
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import Dataset, DataLoader
from src.models.model import GhostFaceNetV2


NUM_CLASSES = 10

LABEL_MAP = {
    f"student{i}": i - 1
    for i in range(1, NUM_CLASSES + 1)
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}


class FaceDataset(Dataset):
    def __init__(self, data_dir, allowed_students=None):
        self.data_dir = Path(data_dir)
        self.image_paths = []
        self.labels = []

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}"
            )

        for person_dir in sorted(self.data_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person = person_dir.name

            if person not in LABEL_MAP:
                continue

            if allowed_students is not None and person not in allowed_students:
                continue

            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                self.image_paths.append(image_path)
                self.labels.append(LABEL_MAP[person])

        if not self.image_paths:
            raise ValueError(
                f"No supported images found in {self.data_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


class FaceClient(fl.client.NumPyClient):
    def __init__(self, data_dir, allowed_students):
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.lr = config["client"]["learning_rate"]
        self.mu = config["client"]["mu"]
        self.batch_size = config["client"]["batch_size"]
        self.epochs = config["client"]["epochs"]
        self.device = torch.device("cpu")

        self.dataset = FaceDataset(
            data_dir,
            allowed_students=allowed_students,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model = GhostFaceNetV2(num_classes=NUM_CLASSES).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        print(f"Training directory: {data_dir}")
        print(f"Assigned students: {sorted(allowed_students)}")
        print(f"Training images: {len(self.dataset)}")

    def get_parameters(self, config):
        return [
            tensor.detach().cpu().numpy()
            for tensor in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()

        if len(parameters) != len(state_dict):
            raise ValueError("Unexpected number of model tensors.")

        updated_state = {
            key: torch.as_tensor(value, dtype=state_dict[key].dtype)
            for key, value in zip(state_dict.keys(), parameters)
        }

        self.model.load_state_dict(updated_state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        global_params = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

        for epoch in range(self.epochs):
            total_loss = 0.0

            for images, labels in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if self.mu > 0:
                    proximal_loss = torch.zeros((), device=self.device)

                    for name, parameter in self.model.named_parameters():
                        proximal_loss += torch.sum(
                            (parameter - global_params[name]) ** 2
                        )

                    loss = loss + (self.mu / 2.0) * proximal_loss

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"loss={total_loss / len(self.loader):.4f}"
            )

        return self.get_parameters(config), len(self.dataset), {}

    def evaluate(self, parameters, config):
        # This is local training-data evaluation, not final test accuracy.
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0
        avg_loss = total_loss / total if total else 0.0

        print(f"Local training-data accuracy: {accuracy:.4f}")

        return avg_loss, total, {"accuracy": accuracy}



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.training.client client1 OR client2")
        sys.exit(1)

    client_id = sys.argv[1].replace("\\", "/").rstrip("/").split("/")[-1]

    if client_id == "client1":
        data_dir = "data/federated_balanced/client1"
    elif client_id == "client2":
        data_dir = "data/federated_balanced/client2"
    else:
        raise ValueError("Client must be client1 or client2.")

    client = FaceClient(data_dir, allowed_students=set(LABEL_MAP.keys()))

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )
