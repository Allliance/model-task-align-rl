"""Deadlines that a task checker's broad Exception handler cannot misclassify."""

import contextlib
import signal
import time


class TaskDeadlineExceeded(BaseException):
    pass


@contextlib.contextmanager
def task_deadline(seconds=2):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    expired = False
    def timeout(*_):
        nonlocal expired
        expired = True
        raise TaskDeadlineExceeded("Task answer extraction/verification deadline exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        if expired:
            # A legacy bare `except:` may swallow even BaseException.
            raise TaskDeadlineExceeded("Task checker swallowed its deadline")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
