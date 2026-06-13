# Asynchronous Gradient Server

Implementación de **Async SGD** y **Ringmaster ASGD** de:

> [Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity](https://arxiv.org/abs/2501.16168)

## Variantes soportadas

```text
Worker:
    recibe model x^(k-delta) y shard
    calcula gradiente local
    envía gradiente al servidor

Server:
    staleness = k_current - iter_sent

    if staleness <= max_staleness:
        gamma = lr

        if use_lr_decay:
            gamma = lr / (1 + staleness)

        x^(k+1) = x^k - gamma * grad
        k = k + 1

    envía el modelo actual y el siguiente shard al worker
```

### Configuración

```text
max_staleness = ∞
    Accepta todos los gradientes (Async SGD).

max_staleness = R
    Descarta gradientes con staleness > R (Ringmaster ASGD).

use_lr_decay = False
    Tasa de aprendizaje constante.

use_lr_decay = True
    gamma = lr / (1 + staleness)
```

| Variante | `max_staleness` | `use_lr_decay` |
|----------|-----------------|--------------|
| Async SGD | `∞` | `False` |
| Ringmaster ASGD | `R` | `False` |
| Decayed Async SGD | `∞` | `True` |
| Decayed Ringmaster ASGD *(experimental)* | `R` | `True` |

> **Note:** La disminución de la tasa de aprendizaje basada en la obsolescencia del gradiente es una extensión experimental de esta implementación y no forma parte del algoritmo Ringmaster ASGD original.
