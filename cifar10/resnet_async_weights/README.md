# CIFAR10 - ResNet18 Async Weights

ResNet-18 custom + AsyncWeightsServer para CIFAR-10.

## Modelo
- ResNet-18 implementado desde cero
- BasicBlock x2 por stage
- Input: 32x32x3 → Output: 10 clases


## Uso

Servidor:
```bash
python -m cifar10.resnet_async_weights.train --epochs 10 --lr 0.01 --batch-size 128 --save "results"
```

Worker:
```bash
python -m cifar10.resnet_async_weights.train --worker --host localhost --save "results"
```

## Parametros
- epochs: 50
- lr: 0.01 (SGD momentum=0.9)
- gamma: 0.5
- batch_size: 128
- shard_size: 2048
- max_staleness: 10
- weight_decay: 5e-4
