import pickle
import socket

from .logger import log


def pickle_code(obj):
    """Codifica un objeto en bytes usando pickle."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def send_raw(sock, raw: bytes):
    """Envía una cadena de bytes sin procesar al socket."""
    sock.sendall(len(raw).to_bytes(4, "big") + raw)


def send_msg(sock, obj):
    """Envía un objeto serializado al socket usando pickle."""
    raw = pickle_code(obj)
    send_raw(sock, raw)


def send_safe(wid, sock, msg, is_raw=False):
    """Envía un mensaje al socket de forma segura, registrando errores."""
    try:
        if is_raw:
            send_raw(sock, msg)
        else:
            send_msg(sock, msg)

        return None
    except Exception as e:
        log.warning(f"Worker {wid} error in send: {e}")
        return wid


def recv_msg(sock):
    """Recibe un mensaje del socket, primero leyendo la longitud."""
    header = recv_exact(sock, 4)
    length = int.from_bytes(header, "big")
    raw = recv_exact(sock, length)
    return pickle.loads(raw)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Recibe exactamente n bytes del socket, levantando una excepción si se cierra prematuramente."""
    buf = bytearray()

    while len(buf) < n:
        chunk = sock.recv(n - len(buf))

        if not chunk:
            raise ConnectionError("Socket cerrado prematuramente")

        buf.extend(chunk)

    return bytes(buf)
