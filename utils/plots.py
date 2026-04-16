from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    conf_matrix, save_path: str | None = None, class_names: List[str] | None = None
):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    num_classes = conf_matrix.shape[0]
    label_position = list(range(num_classes))
    labels = label_position if class_names is None else class_names

    ax.set(
        xticks=label_position,
        yticks=label_position,
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="MNIST Confusion Matrix",
    )

    threshold = conf_matrix.max().item() / 2 if conf_matrix.numel() else 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = int(conf_matrix[i, j].item())

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png")
    else:
        plt.show()


def plot_grid(
    history: List[Union[List[float], tuple]],
    labels: List[str | List[str] | tuple],  # titulo o (titulo, label1, label2, ...)
    n_cols: int | None = None,
    save_path: str | None = None,
):
    if not history:
        return

    n_metrics = len(labels)
    n_epochs = len(history)
    columns = list(zip(*history))

    if len(columns) != n_metrics:
        raise ValueError(
            f"El número de columnas en history ({len(columns)}) no coincide con labels ({n_metrics})"
        )

    # auto rows
    n_cols = n_cols or n_metrics
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = np.array(axs).flatten()

    for idx, (col_values, label) in enumerate(zip(columns, labels)):
        ax = axs[idx]
        title = label if isinstance(label, str) else label[0]

        ax.plot(range(1, n_epochs + 1), col_values)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True)

        if isinstance(label, tuple):
            try:
                ax.legend([label[1], label[2]])
            except IndexError:
                pass

    # Ocultar subplots sobrantes si el grid tiene más celdas que métricas
    for idx in range(n_metrics, len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/grid.png")
    else:
        plt.show()
