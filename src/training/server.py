import os
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import yaml

from src.models.model import GhostFaceNetV2


NUM_CLASSES = 10


class SavingFedProx(fl.server.strategy.FedProx):
    """FedProx strategy that retains the latest aggregated parameters."""

    def __init__(self, *args, initial_parameters, **kwargs):
        super().__init__(
            *args,
            initial_parameters=initial_parameters,
            **kwargs,
        )

        self.final_parameters = initial_parameters

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated is not None:
            self.final_parameters = aggregated

        return aggregated, metrics


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = GhostFaceNetV2(num_classes=NUM_CLASSES)

    initial_arrays = [
        tensor.detach().cpu().numpy()
        for tensor in model.state_dict().values()
    ]

    initial_parameters = fl.common.ndarrays_to_parameters(
        initial_arrays
    )

    strategy = SavingFedProx(
        fraction_fit=config["server"]["fraction_fit"],
        fraction_evaluate=config["server"]["fraction_evaluate"],
        min_fit_clients=config["server"]["min_fit_clients"],
        min_evaluate_clients=config["server"]["min_evaluate_clients"],
        min_available_clients=config["server"]["min_available_clients"],
        proximal_mu=config["server"]["proximal_mu"],
        initial_parameters=initial_parameters,
    )

    print("Starting Federated Learning server...")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Communication rounds: {config['server']['num_rounds']}")

    fl.server.start_server(
        server_address=config["server"]["address"],
        config=fl.server.ServerConfig(
            num_rounds=config["server"]["num_rounds"]
        ),
        strategy=strategy,
    )

    checkpoint_dir = Path(
        config["experiment"]["checkpoints_dir"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_arrays = fl.common.parameters_to_ndarrays(
        strategy.final_parameters
    )

    state_dict = model.state_dict()

    if len(final_arrays) != len(state_dict):
        raise ValueError("Final parameter count does not match the model.")

    final_state = {
        key: torch.as_tensor(
            value,
            dtype=state_dict[key].dtype,
        )
        for key, value in zip(state_dict.keys(), final_arrays)
    }

    model.load_state_dict(final_state)

    checkpoint_path = checkpoint_dir / "final_model.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "label_map": {
                f"student{i}": i - 1
                for i in range(1, NUM_CLASSES + 1)
            },
        },
        checkpoint_path,
    )

    print(f"Saved final global model to: {checkpoint_path}")


if __name__ == "__main__":
    main()