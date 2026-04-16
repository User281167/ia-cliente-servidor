from .pickle_utils import recv_msg


def recv_ddp(sock):
    return DDPMessage.parse(recv_msg(sock))


MSG_WEIGHTS = "weights"
MSG_STEP = "step"
MSG_RESULT = "result"
MSG_ASSIGN = "assign"
MSG_DONE = "done"
MSG_READY = "ready"


class DDPMessage:
    @staticmethod
    def weights(state):
        return {"type": MSG_WEIGHTS, "payload": state}

    @staticmethod
    def step(epoch):
        return {"type": MSG_STEP, "epoch": epoch}

    @staticmethod
    def result(grads, loss, n):
        return {
            "type": MSG_RESULT,
            "payload": {"grads": grads, "loss": loss, "n_samples": n},
        }

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
