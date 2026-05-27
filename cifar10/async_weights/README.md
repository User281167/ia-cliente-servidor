## Cifar 10 DDP async con delta de pesos

Worker entrena localmente y envia `delta = w_local - w_global`.

Servidor aplica delta asyncrono:

`w <- w + gamma * delta`

con correccion por staleness.

### Uso
```bash
# servidor
python -m cifar10.async_weights.train --epochs 10 --conv --normalize --rgb --save "results"

# worker
python -m cifar10.async_weights.train --worker --host localhost --save "results"
```

### Parametros servidor
- `--lr`
- `--epochs`
- `--test-each`
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
