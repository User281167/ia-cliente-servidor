## Tiny ImageNet — MobileNetV2 async SGD por gradientes

[Algoritmo #1 de Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays](https://arxiv.org/abs/2206.07638)

Entrenamiento asíncrono con MobileNetV2 preentrenado en ImageNet.

- Worker recibe pesos globales + shard de datos
- Acumula gradientes (forward + backward, sin optimizer.step)
- Envía gradientes al servidor
- Servidor aplica gradiente con tasa corregida por staleness: `lr / (1 + δ)`
- Servidor ejecuta test cada `test_each` iteraciones

El transform de MobileNet redimensiona las imágenes 64x64 → 224x224.

### Uso

```bash
# servidor
python -m tiny_imagenet.mobilenet.train --epochs 20 --lr 0.01 --save "results"

# worker
python -m tiny_imagenet.mobilenet.train --worker --host localhost --save "results"
```

### Parámetros para el servidor

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--lr` | 0.01 | Tasa de aprendizaje |
| `--epochs` | 20 | Épocas totales |
| `--test-each` | 10 | Iteraciones entre tests |
| `--shard-size` | 5000 | Tamaño del shard por worker |
| `--batch-size` | 128 | Batch local del worker |
| `--max-staleness` | 10 | Máxima diferencia de iteración permitida |
| `--min-workers` | 1 | Workers mínimos para empezar |
| `--port` | 9090 | Puerto de escucha |
| `--save` | None | Carpeta para métricas y reportes |

### Parámetros para el worker

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--worker` | — | Ejecutar como worker |
| `--host` | 0.0.0.0 | IP del servidor |
| `--port` | 9090 | Puerto del servidor |
| `--save` | None | Carpeta para métricas locales |
