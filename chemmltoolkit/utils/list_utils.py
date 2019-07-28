from collections.abc import Iterable


def flatten(l):
    def _flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
    return list(_flatten(l))


def one_hot(feature, tokens):
    return [int(feature == token) for token in tokens]
