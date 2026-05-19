from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass


@dataclass
class ShardAssignment:
    epoch: int
    seed: int
    shard_idx: int  # índice lógico dentro de la época
    start: int  # offset en el dataset
    length: int  # cuántos samples
    batch_size: int  # tamaño del batch


class ShardScheduler:
    """
    Gestiona la cola de shards pendientes de forma thread-safe.

    Cada época tiene max_batches slots. El scheduler emite
    ShardAssignment a los workers que los soliciten. Cuando se
    agotan los shards de una época, avanza automáticamente a la
    siguiente y reconstruye la cola.

    _in_flight: batch_idx -> worker_id: batches del dataset en ejecución
    Permite detectar si un batch ya fue asignado a alguien.
    """

    def __init__(self, data_len: int, shard_size: int, batch_size: int):
        self.data_len = data_len
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.max_batches = data_len // batch_size
        self.max_shards = data_len // shard_size
        self.seed = secrets.randbits(32)

        self._lock = threading.Lock()
        self.current_epoch: int = 0

        # cola de indices de shard pendientes para la época actual
        self._pending: list[int] = list(range(self.max_shards))

        # shard_idx -> worker_id  (qué worker tiene este shard en ejecución)
        self._in_flight: dict[int, int] = {}

    def next_shard(self, wid: int) -> ShardAssignment:
        """
        Devuelve el siguiente ShardAssignment para el worker wid.
        Thread-safe. Avanza de época si se agotaron los shards.
        """
        with self._lock:
            # avanzar época si se agotó la cola
            if not self._pending:
                self.current_epoch += 1
                self._pending = list(range(self.max_shards))
                self._in_flight.clear()  # nueva época asignaciones anteriores caducan
                self.seed = secrets.randbits(32)

            shard_idx = self._pending.pop(0)

            # no debería estar en ejecución si la cola es correcta
            if shard_idx in self._in_flight:
                raise RuntimeError(
                    f"Shard {shard_idx} ya está en ejecución "
                    f"(worker {self._in_flight[shard_idx]})"
                )

            self._in_flight[shard_idx] = wid

            start = shard_idx * self.shard_size
            length = min(self.shard_size, self.data_len - start)

            return ShardAssignment(
                epoch=self.current_epoch,
                seed=self.seed,
                shard_idx=shard_idx,
                start=start,
                length=length,
                batch_size=self.batch_size,
            )

    def complete(self, wid: int, shard_idx: int) -> None:
        """
        Marca el shard como completado y lo elimina de _in_flight.
        Llamar desde _handle_result cuando llega el resultado del worker.
        """
        with self._lock:
            assigned_wid = self._in_flight.pop(shard_idx, None)

            if assigned_wid is None:
                # puede pasar si la época ya avanzó y se limpió _in_flight
                return

            if assigned_wid != wid:
                # batch reasignado (worker lento de época anterior) — ignorar
                pass

    def requeue(self, wid: int) -> None:
        """
        Re-encola todos los shards en ejecución del worker wid.
        Llamar cuando un worker muere inesperadamente.
        """
        with self._lock:
            to_requeue = [bidx for bidx, w in self._in_flight.items() if w == wid]

            for bidx in to_requeue:
                del self._in_flight[bidx]
                self._pending.append(bidx)  # al final de la cola
