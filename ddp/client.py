from __future__ import annotations

import socket
import time
from abc import ABC
from typing import Optional

from .logger import log
from .message import MSG_ASSIGN, MSG_DONE, DDPMessage, recv_ddp
from .pickle_utils import recv_msg, send_msg


class HandlerRegistry:
    def __init__(self):
        self._handlers = {}

    def on(self, msg_type):
        def decorator(fn):
            self._handlers[msg_type] = fn
            return fn

        return decorator

    def get(self, msg_type):
        return self._handlers.get(msg_type)


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

        self._handlers = HandlerRegistry()

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
                msg: dict = DDPMessage.ready(self._worker_id)
                send_msg(sock, msg)

                # Recibir asignación
                assign = recv_ddp(sock)
                DDPMessage.expect(assign, MSG_ASSIGN)

                self._worker_id = assign["meta"]["worker_id"]
                self._assignment = assign["payload"]
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

    # eventos
    def on(self, msg_type):
        return self._handlers.on(msg_type)

    def _loop(self) -> None:
        """
        Loop principal de mensajes.
        `handlers` es un dict  tipo -> callable(msg) → None
        El callable puede enviar de vuelta al servidor con send_msg(self._sock, …)
        """
        while True:
            try:
                msg = DDPMessage.parse(recv_msg(self._sock))
                mtype = msg["type"]

                if mtype == MSG_DONE:
                    log.info("Servidor indicó fin")
                    break

                handler = self._handlers.get(mtype)

                if handler:
                    handler(msg)
                else:
                    log.warning(f"Mensaje desconocido: {mtype}")
            except ValueError as e:
                log.warning(f"Mensaje inválido: {e}")
            except (ConnectionError, OSError) as e:
                log.warning(f"Conexión perdida: {e}")
                self._reconnect()

    def run(self):
        self.connect()
        self._loop()
