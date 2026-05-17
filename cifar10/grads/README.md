## Cifar 10 Datos distribuidos DDP (Promedio de gradientes)

Entrenamiento con mini batches.

- Worker recibe shard (cantidad de datos a tomar)
- En cada shard realiza fordward, backward
- Envia gradientes al servidor

Los Workers realizan el test.

## CIFAR-10 Distributed Data Parallel (DDP)

```bash
# servidor
python -m cifar10.grads.train --epoch 10 --conv --normalize --rgb --save "results"

# worker
python -m cifar10.grads.train --worker --host localhost --save "results"
```

### Parámetros para el servidor
- --lr 
- --epochs
- --batch-size
- --min-workers
- --port
- --rgb: Usar los tres canales de color
- --normalize: Normalizar los valores entre [-1, 1]
- --conv: Usar modelo convolucional simple
- --save: Carpeta donde se guardarán las métricas

### Parámetros para el worker
- --worker
- --host: Dirección IP del servidor
- --port: Puerto de escucha del servidor
- --save: Carpeta donde se guardarán las métricas
