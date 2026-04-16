import time
from functools import wraps


def format_elapse(elapsed: float) -> str:
    if elapsed < 1:
        formatted = f"{elapsed * 1000:.2f} ms"
    elif elapsed < 60:
        formatted = f"{elapsed:.2f} s"
    elif elapsed < 3600:
        mins, secs = divmod(elapsed, 60)
        formatted = f"{int(mins)}m {secs:.2f}s"
    else:
        hours, remainder = divmod(elapsed, 3600)
        mins, secs = divmod(remainder, 60)
        formatted = f"{int(hours)}h {int(mins)}m {secs:.2f}s"

    return formatted


def time_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        formatted = format_elapse(elapsed)

        print(f"{func.__name__} took {formatted}")
        return result

    return wrapper
