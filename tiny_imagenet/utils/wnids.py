from typing import Dict


def load_tiny_imagenet_classes() -> Dict[str, str]:
    path = __file__.replace("wnids.py", "wnids.txt")
    classes = {}

    with open(path) as f:
        for line in f:
            wnid, name = line.strip().split("|")
            classes[wnid] = name

    return classes


tiny_imagenet_classes = load_tiny_imagenet_classes()
