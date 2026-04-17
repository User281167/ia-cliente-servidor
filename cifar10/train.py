import argparse

from .client import CIFAR10Worker
from .server import CIFAR10Server


def run_server(
    gray: bool = False,
    normalize: bool = False,
    conv: bool = False,
    epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 128,
    min_workers: int = 1,
    host: str = "0.0.0.0",
    port: int = 9090,
    save_path: str | None = None,
):
    server = CIFAR10Server(
        gray=gray,
        normalize=normalize,
        conv=conv,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        min_workers=min_workers,
    )

    try:
        server.run(host=host, port=port)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
    finally:
        server.stop_server()
        server.results(save_path=save_path)


def run_client(host, port, save_path=None):
    client = CIFAR10Worker(host, port)

    try:
        client.run()
    except Exception as e:
        print(f"Error al iniciar el cliente: {e}")
    finally:
        client.close()

        if save_path:
            client.save_metrics(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min_workers", type=int, default=1)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--save", type=str, default=None)

    args = parser.parse_args()

    if args.worker:
        run_client(
            args.host,
            args.port,
            save_path=args.save,
        )
    else:
        run_server(
            gray=not args.rgb,
            normalize=args.normalize,
            conv=args.conv,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            min_workers=args.min_workers,
            save_path=args.save,
        )
