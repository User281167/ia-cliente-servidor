from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_confusion_matrix(
    conf_matrix: torch.Tensor | np.ndarray,
    save_path: str | None = None,
    class_names: list[str] | None = None,
):
    """
    Gráfica una matriz de confusión mostrando:
    - Conteo absoluto
    - Porcentaje por fila (clase real)

    Args:
        conf_matrix: Matriz de confusión.
        save_path: Ruta para guardar la imagen (opcional).
        class_names: Lista de nombres de clases (opcional).
    """
    # Convertir a float para normalización
    if isinstance(conf_matrix, torch.Tensor):
        conf_matrix = conf_matrix.float()
        conf_matrix_np = conf_matrix.cpu().numpy()
    else:
        conf_matrix_np = conf_matrix.astype(float)

    # Normalización por filas (porcentaje)
    row_sums = conf_matrix_np.sum(axis=1, keepdims=True)
    conf_matrix_norm = conf_matrix_np / np.clip(row_sums, 1e-8, None)

    # Plot base
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_matrix_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    num_classes = conf_matrix_np.shape[0]
    label_position = list(range(num_classes))
    labels = label_position if class_names is None else class_names

    ax.set(
        xticks=label_position,
        yticks=label_position,
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    # Texto en cada celda (conteo + porcentaje)
    for i in range(num_classes):
        for j in range(num_classes):
            value = int(conf_matrix_np[i, j])
            percentage = conf_matrix_norm[i, j]

            ax.text(
                j,
                i,
                f"{value}\n{percentage:.1%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if percentage > 0.5 else "black",
            )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_grid(
    history: List[Union[List[float], tuple]],
    labels: List[str | List[str] | tuple],  # titulo o (titulo, label1, label2, ...)
    n_cols: int | None = None,
    save_path: str | None = None,
    ax_as_int: bool = True,
    x_label: str = "Epoch",
):
    """
    Graficar varias métricas en una cuadrícula.

    Args:
        history (List[Union[List[float], tuple]]): Lista de listas o tuplas con los valores de las métricas.
            [(val1, val2), () ...] valores en una misma gráfica

        labels (List[str | List[str] | tuple]): Lista de etiquetas para las métricas.
            [(titulo, label1, label2, ...), ...] titulo del plot y legend de cada linea

        n_cols (int | None, optional): Número de columnas en la cuadrícula. Defaults to None.
        save_path (str | None, optional): Ruta para guardar la figura. Defaults to None.
        ax_as_int (bool, optional): Si True, el eje x solo muestra valores enteros. Defaults to True.
    """
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
        ax.set_xlabel(x_label)
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

    # eje x solo entero
    if ax_as_int:
        for ax in axs:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if save_path:
        plt.savefig(f"{save_path}/grid.png")
    else:
        plt.show()
