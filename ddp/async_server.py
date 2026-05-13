from __future__ import annotations

import threading

from .handler_register import HandlerRegistry
from .logger import log
from .message import MSG_READY, DDPMessage, recv_ddp
from .pickle_utils import send_msg
from .server import DDPServer


class DDPAsyncServer(DDPServer):
    """
    Servidor DDP asíncrono uso de handlers para ejecutar peticiones de los workers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._result_queue: list = []
        self._queue_lock = threading.Lock()

        self._handlers = HandlerRegistry()
        self._handlers.on("result")(self._handle_result)

    # ------------------------------------------------------------------
    # Hooks para que las subclases sobreescriban sin tocar el loop
    # ------------------------------------------------------------------

    def _handle_result(self, msg: dict) -> None:
        """
        Subclases sobreescriben esto para aplicar gradientes y devolver
        los nuevos pesos+step+batch al worker.

        La implementación base solo guarda en la cola (sin reenvío).
        """
        wid = msg["worker_id"]
        payload = msg["payload"]

        with self._queue_lock:
            self._result_queue.append((wid, payload))

    # ------------------------------------------------------------------
    # Ciclo de comunicación con cada worker
    # ------------------------------------------------------------------

    def _loop_worker_socket(self, sock, wid: int) -> None:
        """
        Hilo dedicado por worker. Solo maneja dos tipos de mensaje:
          - "result"  → delega a _handle_result (que a su vez envía el
                        siguiente weights+step+batch_assignment)
          - "done"    → cierra el hilo

        Cualquier otro tipo pasa por HandlerRegistry (p.ej. "metrics").
        """
        while True:
            try:
                msg = recv_ddp(sock)
                mtype = msg["type"]

                if mtype == "done":
                    log.info(f"Worker {wid} finalizó")
                    break

                handler = self._handlers.get(mtype)
                if handler:
                    handler(msg)
                else:
                    log.warning(f"Worker {wid} envió mensaje desconocido: {mtype}")

            except (ConnectionError, OSError):
                log.warning(f"Conexión con worker {wid} perdida")
                break
            except ValueError as e:
                log.warning(f"Mensaje inválido de worker {wid}: {e}")

        self._remove_dead([wid])

    def _handshake_worker(self, conn, addr) -> None:
        try:
            msg = recv_ddp(conn)

            try:
                DDPMessage.expect(msg, MSG_READY)
            except ValueError:
                conn.close()
                return

            existing_id = msg.get("worker_id")

            with self._workers_lock:
                if existing_id is not None and existing_id in self._assignments:
                    wid = existing_id
                    log.info(f"Worker {wid} reconectado desde {addr}")
                else:
                    wid = self._next_worker_id
                    self._next_worker_id += 1
                    log.info(f"Nuevo worker {wid} desde {addr}")

                    threading.Thread(
                        target=self._loop_worker_socket,
                        args=(conn, wid),
                        daemon=True,
                    ).start()

                self._workers[wid] = conn

                if len(self._workers) >= self.min_workers:
                    self._ready_event.set()

            assign = self._assignments.get(wid, {})
            send_msg(conn, DDPMessage.assign(wid, **assign))

            if self.worker_config:
                send_msg(conn, DDPMessage.config(**self.worker_config))

        except Exception as e:
            log.warning(f"Error en handshake con {addr}: {e}")
            conn.close()

    # eventos
    def on(self, msg_type):
        """
        Decorador para registrar un handler para un tipo de mensaje específico.

        ej:
            @self.on(MSG_ASSIGN)
            def handle_assign(msg):
                print(msg)
        """
        return self._handlers.on(msg_type)
