from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod
from typing import Optional

from .logger import log
from .pickle_utils import recv_msg, send_msg


class DDPClient(ABC):
    """
    Base para el lado cliente (worker) del entrenamiento de datos distribuido.

    Gestiona:
      - Conexión / reconexión TCP
      - Loop de mensajes
      - Despacho a handlers por tipo de mensaje

    Subclases implementan `run()` que define qué hacer con cada mensaje.
    """

    RECONNECT_DELAY = 3  # segundos entre intentos de reconexión
    RECONNECT_ATTEMPTS = 20  # intentos antes de rendirse

    def __init__(self, server_host: str, server_port: int):
        self.server_host = server_host
        self.server_port = server_port

        self._sock: Optional[socket.socket] = None
        self._worker_id: Optional[int] = None
        self._assignment: dict = {}
        self._connected = False

    def connect(self) -> None:
        """Conecta al servidor y recibe la asignación inicial (handshake)."""
        for attempt in range(1, self.RECONNECT_ATTEMPTS + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                # Reuse address
                # Enable keepalive
                # para evitar desconexiones por inactividad o periodo de entrenamiento
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.connect((self.server_host, self.server_port))
                self._sock = sock

                # Enviar ready (con worker_id si es reconexión)
                msg: dict = {"type": "ready"}

                if self._worker_id is not None:
                    msg["worker_id"] = self._worker_id

                send_msg(sock, msg)

                # Recibir asignación
                assign = recv_msg(sock)

                if assign.get("type") != "assign":
                    raise ValueError(
                        f"Se esperaba 'assign', se recibio {assign.get('type')}"
                    )

                self._worker_id = assign["worker_id"]
                self._assignment = {
                    k: v for k, v in assign.items() if k not in ("type", "worker_id")
                }
                self._connected = True

                log.info(
                    f"Conectado como worker {self._worker_id} | "
                    f"asignación: {self._assignment}"
                )
                return

            except Exception as e:
                log.warning(f"Intento {attempt}/{self.RECONNECT_ATTEMPTS} fallido: {e}")
                time.sleep(self.RECONNECT_DELAY)

        raise ConnectionError(
            f"No se pudo conectar a {self.server_host}:{self.server_port} "
            f"tras {self.RECONNECT_ATTEMPTS} intentos"
        )

    def close(self) -> None:
        """Cierra el socket limpiamente."""
        self._connected = False

        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
            except Exception:
                pass

            self._sock = None

        log.info("Cliente cerrado")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _reconnect(self) -> None:
        self.close()
        log.info("Intentando reconexión…")
        self.connect()

    def _loop(self, handlers: dict) -> None:
        """
        Loop principal de mensajes.
        `handlers` es un dict  tipo -> callable(msg) → None
        El callable puede enviar de vuelta al servidor con send_msg(self._sock, …)
        """
        while True:
            try:
                msg = recv_msg(self._sock)
                mtype = msg.get("type")

                if mtype == "done":
                    log.info("Servidor indicó fin — cerrando")
                    break

                if mtype == "assign":
                    # re-asignación durante reconexión
                    self._assignment = {
                        k: v for k, v in msg.items() if k not in ("type", "worker_id")
                    }
                    log.info(f"Re-asignación recibida: {self._assignment}")
                    continue

                handler = handlers.get(mtype)

                if handler:
                    handler(msg)
                else:
                    log.warning(f"Mensaje desconocido ignorado: {mtype}")

            except (ConnectionError, OSError) as e:
                log.warning(f"Conexión perdida: {e}")

                try:
                    self._reconnect()
                except ConnectionError:
                    log.error("No se pudo reconectar — terminando cliente")
                    break

    @abstractmethod
    def run() -> None:
        """Blocking loop — escucha mensajes y ejecuta funciones."""
        ...
