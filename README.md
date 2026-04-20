# Servidor de parámetros

## Instalación con `uv`
```bash
uv sync
```

## Instalacion con `pip`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Cifar 10 Datos distribuidos DDP

Servidor de parametros y worker
- Uso de TCP/IP para la comunicación de eventos y datos
- Uso de pytorch para modelo de entrenamiento
- Pickle como serialización para el envio de mensajes

## CIFAR-10 Distributed Data Parallel (DDP)

```bash
# servidor
python -m cifar10.train --epoch 10 --conv --normalize --rgb --save "results"

# worker
python -m cifar10.train --worker --host localhost --save "results"
```
