import argparse

from .client import CIFAR10Worker
from .server import CIFAR10Server


def run_server(
    gray: bool = False,
    normalize: bool = False,
    conv: bool = False,
    epochs: int = 20,
    lr: float = 0.001,
    shard_size: int = 5000,
    batch_size: int = 128,
    max_staleness: int = 10,
    test_each: int = 10,
    min_workers: int = 1,
    host: str = "0.0.0.0",
    port: int = 9090,
    save_path: str | None = None,
    use_lr_decay: bool = False,
):
    server = CIFAR10Server(
        gray=gray,
        normalize=normalize,
        conv=conv,
        epochs=epochs,
        lr=lr,
        shard_size=shard_size,
        batch_size=batch_size,
        max_staleness=max_staleness,
        test_each=test_each,
        min_workers=min_workers,
        save_path=save_path,
        use_lr_decay=use_lr_decay,
    )

    try:
        server.run(host=host, port=port)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
    finally:
        server.stop_server()


def run_client(host, port, save_path=None):
    client = CIFAR10Worker(host, port, save_path)

    try:
        client.run()
    except Exception as e:
        print(f"Error al iniciar el cliente: {e}")
    finally:
        client.close()

        if save_path:
            client.save_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--use-lr-decay", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-workers", type=int, default=1)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-staleness",
        type=int,
        default=None,
        help="Si un worker responde despues de max-staleness, se descarta su gradiente",
    )
    parser.add_argument("--test-each", type=int, default=10)

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
            shard_size=args.shard_size,
            batch_size=args.batch_size,
            max_staleness=args.max_staleness,
            test_each=args.test_each,
            min_workers=args.min_workers,
            save_path=args.save,
            host=args.host,
            port=args.port,
            use_lr_decay=args.use_lr_decay,
        )
