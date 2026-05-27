import os

import pandas as pd
import torch

from sinc import SyncGradServer
from utils import plot_grid


class SyncWeightsServer(SyncGradServer):
    """
    Servidor para el entrenamiento distribuido con delta de pesos.
    """

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 128,
        min_workers: int = 1,
        config: dict | None = None,
        save_path: str | None = None,
        load_model: bool = False,
    ):
        if config is None:
            config = {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
            }

        super().__init__(
            epochs,
            lr,
            batch_size,
            min_workers,
            config,
            save_path,
            load_model,
        )

        self.metrics = pd.DataFrame(
            columns=[
                "workers",
                "worker_res",
                "loss",
                "accuracy",
                "eval_loss",
                "eval_accuracy",
                "delta_norm",
                "elapsed",
            ]
        )

    def results(self):
        """Guarda las métricas en un archivo Excel y genera gráfico de resultados."""
        save_path = self.save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        if save_path:
            self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))

            description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
            description.to_excel(
                os.path.join(save_path, "description_server.xlsx"),
                index=True,
            )

        plot_grid(
            history=[
                (
                    # unir train/test en una sola gráfica
                    (self.metrics["loss"][i], self.metrics["eval_loss"][i]),
                    ((self.metrics["accuracy"][i], self.metrics["eval_accuracy"][i])),
                    self.metrics["delta_norm"][i],
                )
                for i in range(len(self.metrics))
            ],
            labels=[
                ("Loss", "Train", "Test"),
                ("Accuracy", "Train", "Test"),
                "Weights Norm",
            ],
            n_cols=1,
            save_path=save_path,
        )

        # evaluate classification
        acc, _ = self.evaluate_classification()

        # argumentos
        if save_path:
            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"batch_size: {self.batch_size}\n")
                f.write(f"Final accuracy: {acc}")

    def _aggregate(self, results):
        """
        Promedio ponderado de los pesos de los workers.

        FedAvg:
            w_new = w_global + η * Σ (n_i/N)

        results = [{delta, samples}]
        """
        N_total = sum(r["samples"] for r in results)

        accum_delta = {}

        for i in range(len(results)):
            n_i = results[i]["samples"]
            weight = n_i / N_total
            delta_i = results[i]["delta"]

            for k, d in delta_i.items():
                d = torch.as_tensor(d, device=self.device)

                if k not in accum_delta:
                    accum_delta[k] = weight * d
                else:
                    accum_delta[k] += weight * d

        # Norma L2
        dnorm = 0.0

        # Aplicar delta al modelo global
        state = self.model.state_dict()

        for k in state:
            state[k] = state[k] + accum_delta[k]
            dnorm += (accum_delta[k] ** 2).sum().item()

        self.model.load_state_dict(state)

        return dnorm**0.5
