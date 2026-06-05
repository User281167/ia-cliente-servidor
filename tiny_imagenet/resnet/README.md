# ResNet50 para Tiny ImageNet

Implementación mejorada usando ResNet50 preentrenado en ImageNet.

## Mejoras sobre MobileNet V2

1. **Arquitectura más adecuada**: ResNet50 tiene mejor rendimiento en transfer learning para datasets como Tiny ImageNet
2. **Tamaño de imagen correcto**: 224x224 (tamaño nativo del preentrenamiento)
3. **Fine-tuning parcial**: Descongelar solo las ultimas capas (layer3, layer4, fc) para adaptarse a Tiny ImageNet
4. **Clasificador simple**: Solo una capa Linear + Dropout (evita overfitting)
5. **Optimizador AdamW**: Mejor que Adam para regularización
6. **Scheduler OneCycleLR**: Con warmup inicial para mejor convergencia
7. **Batch size reducido**: 64 en lugar de 128 para mejor estabilidad
8. **Test cada 5 epochs**: Menos frecuente que cada 10 para reducir overhead

## Uso

```bash
# Servidor
python -m tiny_imagenet.resnet.train --epochs 20 --batch-size 64 --lr 0.001 --save ./results

# Worker
python -m tiny_imagenet.resnet.train --worker --host localhost --save ./results
```

## Parametros recomendados

- `--lr`: 0.001 (learning rate base)
- `--batch-size`: 64 (equilibrio entre velocidad y estabilidad)
- `--epochs`: 20-30
- `--test-each`: 5 (evaluar cada 5 epochs)
- `--shard-size`: 5000 (tamaño de cada shard para workers)
- `--port`: 9090
