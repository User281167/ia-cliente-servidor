## Cifar 10 DDP async con delta de pesos

Worker entrena localmente y envia `delta = w_local - w_global`.

Servidor aplica delta asyncrono:

`w <- w + gamma * delta`

con correccion por staleness.

### Uso
```bash
# servidor
python -m tiny_imagenet.resnet18.async.train --epoch 10 --batch-size 128 --save "results"

# worker
python -m tiny_imagenet.resnet18.async.train --worker --host localhost --save "results"
```

### Parametros servidor
- `--lr`
- `--epochs`
- `--shard-size`
- `--batch-size`
- `--max-staleness`
- `--port`
- `--rgb`
- `--normalize`
- `--conv`
- `--save`

### Parametros worker
- `--worker`
- `--host`
- `--port`
- `--save`
