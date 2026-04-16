import logging

log = logging.getLogger("servidor-de-paramertros")
log.setLevel(logging.DEBUG)

if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_h)

log.propagate = False
