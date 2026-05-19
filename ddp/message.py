"""
Módulo para la creación y manipulación de mensajes DDP (Distributed Data Parallel).
Permite la creación estandarizada de mensajes para la comunicación en paralelo entre workers y servidor.
Evitando errores comunes como escribir mal el tipo de mensaje o pasar argumentos incorrectos.
"""

from .pickle_utils import recv_msg


def recv_ddp(sock):
    """
    Recibe un mensaje DDP del socket y lo parsea.
    Verifica que el contenido sea válido y que el tipo de mensaje sea reconocido.
    """
    return DDPMessage.parse(recv_msg(sock))


"""
Constantes para los tipos de mensajes DDP.
Mensajes comunes para la comunicación entre workers y servidor en paralelo.
"""
MSG_WEIGHTS = "weights"
MSG_STEP = "step"
MSG_RESULT = "result"
MSG_ASSIGN = "assign"
MSG_DONE = "done"
MSG_READY = "ready"
MSG_CONFIG = "config"


class DDPMessage:
    """
    Clase para la creación y manipulación de mensajes DDP.
    Proporciona métodos estáticos para crear mensajes comunes.
    """

    @staticmethod
    def weights(state):
        return {"type": MSG_WEIGHTS, "payload": state}

    @staticmethod
    def step(epoch, seed=None):
        msg = {"type": MSG_STEP, "epoch": epoch}

        if seed is not None:
            msg["seed"] = seed

        return msg

    @staticmethod
    def result(grads, loss, n):
        return {
            "type": MSG_RESULT,
            "payload": {"grads": grads, "loss": loss, "n_samples": n},
        }

    @staticmethod
    def config(**data):
        return {"type": MSG_CONFIG, "payload": data}

    @staticmethod
    def assign(worker_id, **data):
        return {"type": MSG_ASSIGN, "meta": {"worker_id": worker_id}, "payload": data}

    @staticmethod
    def done():
        return {"type": MSG_DONE}

    @staticmethod
    def ready(worker_id=None):
        return {
            "type": MSG_READY,
            "meta": {"worker_id": worker_id} if worker_id else {},
        }

    @staticmethod
    def is_type(msg, t):
        return msg.get("type") == t

    @staticmethod
    def expect(msg, expected_type):
        t = msg.get("type")

        if t != expected_type:
            raise ValueError(f"Expected {expected_type}, got {t}")

        return msg

    @staticmethod
    def parse(msg):
        if not isinstance(msg, dict):
            raise TypeError("Message must be dict")
        if "type" not in msg:
            raise ValueError("Missing type")

        return msg

    @staticmethod
    def msg(type, **data):
        return {"type": type, "payload": data}
