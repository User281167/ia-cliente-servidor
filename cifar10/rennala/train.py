import argparse

from .client import get_worker, worker_type
from .server import get_server, server_type


def run_server(
    server_type: server_type = "grad",
    B: int = 10,
    gray: bool = False,
    normalize: bool = False,
    conv: bool = False,
    epochs: int = 20,
    lr: float = 0.001,
    gamma: float = 0.1,
    shard_size: int = 5000,
    batch_size: int = 128,
    test_each: int = 10,
    min_workers: int = 1,
    host: str = "0.0.0.0",
    port: int = 9090,
    save_path: str | None = None,
):
    server = get_server(
        server_type=server_type,
        B=B,
        gray=gray,
        normalize=normalize,
        conv=conv,
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        shard_size=shard_size,
        batch_size=batch_size,
        test_each=test_each,
        min_workers=min_workers,
        save_path=save_path,
    )

    try:
        server.run(host=host, port=port)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")


def run_client(worker_type: worker_type, host, port, save_path=None):
    client = get_worker(worker_type, host, port, save_path)

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
    parser.add_argument("--type", type=str, choices=["grad", "weights"], default="grad")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-workers", type=int, default=1)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--B",
        type=int,
        default=10,
        help="Cantidad de resultados que acomula el servidor antes de actualizar",
    )
    parser.add_argument("--test-each", type=int, default=2)
    parser.add_argument("--save", type=str, default=None)

    args = parser.parse_args()

    if args.worker:
        run_client(args.type, args.host, args.port, args.save)
    else:
        run_server(
            server_type=args.type,
            B=args.B,
            gray=not args.rgb,
            normalize=args.normalize,
            conv=args.conv,
            epochs=args.epochs,
            lr=args.lr,
            gamma=args.gamma,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
            test_each=args.test_each,
            min_workers=args.min_workers,
            save_path=args.save,
            host=args.host,
            port=args.port,
        )
