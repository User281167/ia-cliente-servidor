import os

import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from PIL import Image
from torch.utils.data import Subset

from .load_data import TinyImageNetLazy
from .wnids import load_tiny_imagenet_classes


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


def save_class_report(per_class_acc, conf, label_names, save_path):
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
    classes = load_tiny_imagenet_classes()

    for cls in sorted_idx.tolist():
        acc = per_class_acc[cls].item()

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
    for col in ["A", "B", "C", "D", "E", "F"]:
        ws.column_dimensions[col].width = 20

    # columnas de imagen (más anchas)
    ws.column_dimensions["E"].width = 18
    ws.column_dimensions["F"].width = 18

    # -------------------------
    # INSERT IMAGES + ROW HEIGHT
    # -------------------------
    for i, row in df.iterrows():
        excel_row = i + 2  # header offset

        # altura de fila (IMPORTANTE)
        ws.row_dimensions[excel_row].height = 50  # ~64px visual

        # -------- imagen principal --------
        img_path = os.path.join(img_dir, row["img"])
        if os.path.exists(img_path):
            img = XLImage(img_path)
            img.width = 64
            img.height = 64
            ws.add_image(img, f"E{excel_row}")

        # -------- imagen confundida --------
        if row["img_confused"] is not None:
            img_conf_path = os.path.join(img_dir, row["img_confused"])

            if os.path.exists(img_conf_path):
                img2 = XLImage(img_conf_path)
                img2.width = 64
                img2.height = 64
                ws.add_image(img2, f"F{excel_row}")

    # -------------------------
    # SAVE
    # -------------------------
    output_file = os.path.join(save_path, "report.xlsx")
    wb.save(output_file)

    print(f"Excel guardado en: {output_file}")


def excel_report(per_class_acc, conf, loader, save_path):
    """
    Generar un informe de clases en formato Excel.

    Args:
        per_class_acc (Tensor): Tensor con la precisión por clase.
        conf (Tensor | ndarray): Matriz de confusión.
        loader (DataLoader): DataLoader con el dataset.
        save_path (str): Directorio donde se guardará el informe.
    """
    dataset = loader.dataset

    # unwrap Subset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(dataset, TinyImageNetLazy):
        dataset = dataset.hf_dataset
    label_names = dataset.features["label"].names

    img_dir = os.path.join(save_path, "img")

    if not os.path.exists(img_dir):
        extract_images_per_class(img_dir, dataset)

    df = save_class_report(per_class_acc, conf, label_names, save_path)
    export_to_excel(df, save_path, img_dir)
