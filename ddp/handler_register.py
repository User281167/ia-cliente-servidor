from __future__ import annotations


class HandlerRegistry:
    """
    Registro de handlers para mensajes del cliente/server DDP.
    Permite asociar funciones a tipos de mensajes específicos.

    ej:
        @registry.on(MSG_ASSIGN)
        def handle_assign(msg):
            print(msg)
    """

    def __init__(self):
        self._handlers = {}

    def on(self, msg_type):
        def decorator(fn):
            self._handlers[msg_type] = fn
            return fn

        return decorator

    def get(self, msg_type):
        return self._handlers.get(msg_type)
