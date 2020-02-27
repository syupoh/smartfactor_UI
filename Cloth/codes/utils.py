from contextlib import contextmanager


__all__ = ['task']


@contextmanager
def task(_=''):
    yield


class Dummy:
    pass
