import os
import shutil

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image
from torch.utils.data import Subset

from tiny_imagenet.load_data import TinyImageNetLazy
from tiny_imagenet.utils.wnids import tiny_imagenet_classes


def compute_confusion_matrix_and_accuracy(model, loader, num_classes, device=None):
    """
    Calcula matriz de confusión, accuracy global, accuracy por clase
    y top-5 accuracy por clase usando únicamente tensores de PyTorch.

    Args:
        model: Modelo de PyTorch.
        loader: DataLoader.
        num_classes: Número de clases.
        device: 'cpu' o 'cuda'.

    Returns:
        acc: Accuracy global (float)
        conf: Tensor (num_classes, num_classes)
        per_class_acc: Tensor (num_classes,)
        per_class_top5_acc: Tensor (num_classes,)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    correct = 0
    total = 0

    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    per_class_correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    per_class_total = torch.zeros(num_classes, dtype=torch.int64, device=device)
    per_class_top5_correct = torch.zeros(num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            preds = outputs.argmax(dim=1)
            _, pred5 = outputs.topk(5, dim=1)

            # Accuracy global
            correct += (preds == y).sum()
            total += y.size(0)

            # Matriz de confusión
            conf.index_put_(
                (y, preds), torch.ones_like(y, dtype=torch.int64), accumulate=True
            )

            # Total por clase
            per_class_total.index_put_(
                (y,), torch.ones_like(y, dtype=torch.int64), accumulate=True
            )

            # Correctos top-1
            mask_top1 = preds == y
            if mask_top1.any():
                per_class_correct.index_put_(
                    (y[mask_top1],),
                    torch.ones_like(y[mask_top1], dtype=torch.int64),
                    accumulate=True,
                )

            # Correctos top-5
            mask_top5 = (pred5 == y.unsqueeze(1)).any(dim=1)
            if mask_top5.any():
                per_class_top5_correct.index_put_(
                    (y[mask_top5],),
                    torch.ones_like(y[mask_top5], dtype=torch.int64),
                    accumulate=True,
                )

    per_class_acc = per_class_correct.float() / per_class_total.clamp(min=1).float()
    per_class_top5_acc = (
        per_class_top5_correct.float() / per_class_total.clamp(min=1).float()
    )

    acc = (correct.float() / total).item() if total > 0 else 0.0

    return acc, conf, per_class_acc, per_class_top5_acc


def clean_name(name: str):
    return name.split(",")[0].strip().replace(" ", "_")


def extract_images_per_class(save_path, hf_dataset):
    """
    Guardar una imágen por clase en el directorio `save_path`.

    Args:
        save_path (str): Directorio donde se guardarán las imágenes.
        hf_dataset (Dataset): Dataset de Hugging Face con las imágenes y etiquetas.
    """
    os.makedirs(save_path, exist_ok=True)

    label_names = hf_dataset.features["label"].names
    saved = set()

    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]

        label = sample["label"]
        img = sample["image"]

        if label in saved:
            continue

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        class_name = clean_name(label_names[label])

        path = os.path.join(save_path, f"{class_name}.png")

        img.save(path)

        saved.add(label)

        if len(saved) == len(label_names):
            break

    print(f"Guardadas {len(saved)} clases en {save_path}")


def save_class_report(
    per_class_acc, conf, label_names, save_path, per_class_top5_acc=None
):
    """
    Guardar un informe de las clases en un excel.
    Ordenar clases por precisión descendente.

    Args:
        per_class_acc (Tensor): Tensor con la precisión por clase.
        conf (Tensor | ndarray): Matriz de confusión.
        label_names (list): Lista de nombres de las clases.
        save_path (str): Directorio donde se guardará el informe.
    """

    sorted_idx = torch.argsort(per_class_acc, descending=True)

    rows = []
    classes = tiny_imagenet_classes

    for cls in sorted_idx.tolist():
        acc = per_class_acc[cls].item()

        # Obtener top5 accuracy para esta clase
        top5_acc = None
        if per_class_top5_acc is not None:
            top5_acc = float(per_class_top5_acc[cls])

        row_conf = conf[cls].clone()
        row_conf[cls] = 0

        top_conf = None
        if row_conf.sum() > 0:
            top_conf = torch.argmax(row_conf).item()

        class_name = clean_name(label_names[cls])
        top_confused_class = (
            clean_name(label_names[top_conf]) if top_conf is not None else None
        )

        total_errors = conf[cls].sum().item()

        acc_confused = (
            conf[cls][top_conf].item() / total_errors
            if top_conf is not None and total_errors > 0
            else 0
        )

        rows.append(
            {
                "class_name": classes.get(class_name, class_name),
                "accuracy": acc,
                "top5_accuracy": top5_acc,
                "top_confused_class": classes.get(
                    top_confused_class, top_confused_class
                ),
                "acc_confused": acc_confused,
                "img": f"{class_name}.png",
                "img_confused": f"{top_confused_class}.png"
                if top_confused_class
                else None,
            }
        )

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(save_path, "class_report.xlsx"), index=False)

    return df


def export_to_excel(df, save_path, img_dir):
    """
    Cambiar labels por imágenes en el informe de clases.

    Args:
        df (DataFrame): DataFrame con los datos de las clases.
        save_path (str): Directorio donde se guardará el informe.
        img_dir (str): Directorio donde se encuentran las imágenes.
    """

    wb = Workbook()
    ws = wb.active
    ws.title = "report"

    # -------------------------
    # HEADERS
    # -------------------------
    ws.append(list(df.columns))

    # -------------------------
    # DATA ROWS
    # -------------------------
    for _, row in df.iterrows():
        ws.append(list(row.values))

    # -------------------------
    # STYLE: columnas
    # -------------------------
    # columnas A-E
    for col in ["A", "B", "C", "D", "E", "F", "G"]:
        ws.column_dimensions[col].width = 20

    # columnas de imagen (más anchas)
    ws.column_dimensions["F"].width = 18
    ws.column_dimensions["G"].width = 18

    # -------------------------
    # INSERT IMAGES + ROW HEIGHT
    # -------------------------
    for i, row in df.iterrows():
        excel_row = i + 2  # header offset

        # altura de fila (IMPORTANTE)
        ws.row_dimensions[excel_row].height = 50  # ~64px visual

        # -------- imagen principal --------
        img_path = os.path.join(img_dir, str(row["img"]))
        if os.path.exists(img_path):
            img = XLImage(img_path)
            img.width = 64
            img.height = 64
            ws.add_image(img, f"F{excel_row}")

        # -------- imagen confundida --------
        if row["img_confused"] is not None:
            img_conf_path = os.path.join(img_dir, str(row["img_confused"]))

            if os.path.exists(img_conf_path):
                img2 = XLImage(img_conf_path)
                img2.width = 64
                img2.height = 64
                ws.add_image(img2, f"G{excel_row}")

    # -------------------------
    # SAVE
    # -------------------------
    output_file = os.path.join(save_path, "report.xlsx")
    wb.save(output_file)

    print(f"Excel guardado en: {output_file}")


def excel_report(per_class_acc, conf, loader, save_path, per_class_top5_acc=None):
    """
    Generar un informe de clases en formato Excel.

    Args:
        per_class_acc (Tensor): Tensor con la precisión por clase.
        conf (Tensor | ndarray): Matriz de confusión.
        loader (DataLoader): DataLoader con el dataset.
        save_path (str): Directorio donde se guardará el informe.
    """
    dataset = loader.dataset

    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(dataset, TinyImageNetLazy):
        dataset = dataset.hf_dataset

    label_names = dataset.features["label"].names

    img_dir = os.path.join(save_path, "img")

    if not os.path.exists(img_dir):
        extract_images_per_class(img_dir, dataset)

    df = save_class_report(
        per_class_acc, conf, label_names, save_path, per_class_top5_acc
    )
    export_to_excel(df, save_path, img_dir)

    # remove img
    shutil.rmtree(img_dir, ignore_errors=True)
