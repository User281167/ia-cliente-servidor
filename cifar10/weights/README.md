## Cifar 10 Datos distribuidos DDP (Delta de pesos)

Entrenamiento con mini batches.

- Worker recibe shard (cantidad de datos a tomar)
- Divide el shard en batches
- En cada batch realiza fordward, backward y step

Los Workers realizan el test. 

## CIFAR-10 Distributed Data Parallel (DDP)

```bash
# servidor
python -m cifar10.weights.train --epoch 10 --conv --normalize --rgb --save "results"

# worker
python -m cifar10.weights.train --worker --host localhost --save "results"
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
