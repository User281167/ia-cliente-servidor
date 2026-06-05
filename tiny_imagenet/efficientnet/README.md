# EfficientNet-B0 para Tiny ImageNet

Implementación liviana usando EfficientNet-B0 preentrenado en ImageNet.

## Características

- **Modelo liviano**: ~5.3M parámetros totales
- **Parámetros entrenables**: ~1.4M (solo classifier + última capa de features)
- **Fine-tuning parcial**: Solo classifier y features.8 son entrenables
- **Optimizador**: AdamW con weight_decay=5e-4
- **Scheduler**: OneCycleLR con warmup


## Uso

```bash
# Servidor
python -m tiny_imagenet.efficientnet.train --epochs 20 --batch-size 64 --lr 0.001 --save ./results

# Worker
python -m tiny_imagenet.efficientnet.train --worker --host localhost --save ./results
```

## Parámetros recomendados

- `--lr`: 0.001-0.003
- `--batch-size`: 64-128
- `--epochs`: 20-30
- `--test-each`: 5
