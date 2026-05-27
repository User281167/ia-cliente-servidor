import argparse

from .client import Worker
from .server import Server


def run_server(
    epochs: int = 20,
    lr: float = 0.001,
    gamma: float = 0.1,
    shard_size: int = 5000,
    batch_size: int = 128,
    max_staleness: int = 10,
    host: str = "0.0.0.0",
    port: int = 9090,
    save_path: str | None = None,
):
    server = Server(
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        shard_size=shard_size,
        batch_size=batch_size,
        max_staleness=max_staleness,
        save_path=save_path,
    )

    try:
        server.run(host=host, port=port)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")
    finally:
        server.stop_server()


def run_client(host, port, save_path=None):
    client = Worker(host, port)

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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-staleness",
        type=int,
        default=10,
        help="Si un worker responde despues de max-staleness, se descarta su delta",
    )

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
            epochs=args.epochs,
            lr=args.lr,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
            max_staleness=args.max_staleness,
            save_path=args.save,
            host=args.host,
            port=args.port,
        )
