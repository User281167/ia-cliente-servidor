## Cifar 10 Datos distribuidos DDP asíncrono Rennala SGD

### Uso
```bash
# servidor
python -m cifar10.rennala.train --epoch 10 --conv --normalize --rgb --save "results"

# worker
python -m cifar10.rennala.train --worker --host localhost --save "results"
```

### Parámetros para el servidor
- --type [grad|weights]: Tipo de servidor/worker a usar (promedio de gradiente o delta de pesos)
- --lr 
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
