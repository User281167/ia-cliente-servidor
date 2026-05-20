## Tiny ImageNet Distributed Data Parallel (DDP)

Entrenamiento con mini batches.

- Worker recibe shard
- Divide el shard en batches
- En cada batch realiza fordward, backward y step

Los Workers realizan el test y envian los resultados (delta de pesos) al servidor.

## Arquitectura Resnet18 Simple

## Ejecución

```bash
# servidor
python -m tiny_imagenet.resnet18.train --epoch 10 --batch-size 128 --save "results"

# worker
python -m tiny_imagenet.resnet18.train --worker --host localhost --save "results"
```

### Parámetros para el servidor
- --epochs,
- --lr,
- --batch-size,
- --min-workers,
- --save-path,
- --worker-timeout,

### Parámetros para el worker
- --worker
- --host: Dirección IP del servidor
- --port: Puerto de escucha del servidor
- --save: Carpeta donde se guardarán las métricas
