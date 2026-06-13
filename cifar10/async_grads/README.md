## Cifar 10 Datos distribuidos DDP asíncrono con SGD distribuido

[Algoritmo #1 de Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays](https://arxiv.org/abs/2206.07638)

Entrenamiento con gradientes asíncronos

- Worker recibe batch (cantidad de datos a tomar)
- En cada batch realiza fordward, backward
- Envia gradientes al servidor

Los Workers realizan el test con batch


### Uso
```bash
# servidor
python -m cifar10.async_grads.train --epoch 10 --conv --normalize --rgb --save "results"

# worker
python -m cifar10.async_grads.train --worker --host localhost --save "results"
```

### Parámetros para el servidor
- --lr 
- --use-lr-decay: Con tasa de decaimiento del learning rate
- --epochs
- --test-each: Cantidad de iteraciones para el test
- --shard-size: Tamaño de los shards de datos (datos a tomar en el worker)
- --batch-size: Tamaño de los batches de datos (datos a procesar en cada iteración en el worker <= tamaño del shard)
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
