import argparse

from .client import CIFAR10Worker
from .server import CIFAR10Server


def run_server(
    gray: bool = False,
    normalize: bool = False,
    conv: bool = False,
    epochs: int = 20,
    lr: float = 0.001,
    workers: int = 1,
    min_workers: int = 1,
    host: str = "0.0.0.0",
    port: int = 9090,
):
    server = CIFAR10Server(
        gray=gray,
        normalize=normalize,
        conv=conv,
        epochs=epochs,
        lr=lr,
        workers=workers,
        min_workers=min_workers,
    )

    try:
        server.run(host=host, port=port)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
    finally:
        server.stop_server()

        print(server.metrics)


def run_client(host, port, gray=True, normalize=True, lr=0.01):
    client = CIFAR10Worker(host, port, gray=gray, normalize=normalize, lr=lr)

    try:
        client.run()
    except Exception as e:
        print(f"Error al iniciar el cliente: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min_workers", type=int, default=1)

    parser.add_argument("--worker", action="store_true")

    args = parser.parse_args()

    if args.worker:
        run_client(
            args.host,
            args.port,
            gray=not args.rgb,
            normalize=args.normalize,
            lr=args.lr,
        )
    else:
        run_server(
            gray=not args.rgb,
            normalize=args.normalize,
            conv=args.conv,
            epochs=args.epochs,
            lr=args.lr,
            workers=args.workers,
            min_workers=args.min_workers,
        )
