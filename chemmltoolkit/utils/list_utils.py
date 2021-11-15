from collections.abc import Iterable
from collections.abc import MutableMapping
from typing import Any, Dict, List, Tuple


def flatten(l: List[Any]) -> List[Any]:  # noqa: E741
    """Flattens a nested list into a flat list of elements.

    Args:
        l: The list to flatten.

    Returns:
        The flattened list.
    """
    def _flatten(l: List[Any]):  # noqa: E741
        for el in l:
            if _is_iterable(el):
                yield from flatten(el)
            else:
                yield el

    return list(_flatten(l))


def flatten_dict(d: Dict[Any, Any], separator: str = '.') -> Dict[Any, Any]:
    """Flattens a nested dictionary into a flat dictionary.

    This will flatten any nested dictionary items using the specified
    separator, for example if the separator is '.' then in the form
    'grandparent.parent.key'.

    Args:
        d: The dictionary to flatten.
        separator: The string to separate levels of the dictionary.

    Returns:
        The flattened dictionary.
    """
    def _flatten_dict(d: MutableMapping[Any, Any],
                      separator: str,
                      prefix: str) -> List[Tuple[Any, Any]]:
        items: List[Tuple[Any, Any]] = []
        for k, v in d.items():
            new_key = prefix + k
            if isinstance(v, MutableMapping):
                new_prefix = new_key + separator
                items.extend(_flatten_dict(v, separator, new_prefix))
            else:
                items.append((new_key, v))
        return items

    items = _flatten_dict(d, separator, '')
    return dict(items)


def merge_dict(d: Dict[Any, Any], defaults: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merges a dictionary with a set of default values.

    This will combine the entries in two dictionaries, with the first
    argument taking preference.

    Args:
        d: The dictionary of specified values.
        default: The dictionary of default values.

    Returns:
        The merged dictionary.
    """
    return {**defaults, **d}


def one_hot(feature: Any, tokens: List[Any]) -> List[int]:
    """One-hot encodes a feature.

    This will one-hot encode the feature, using a specified list of tokens.

    Args:
        feature: The feature value to one-hot encode.
        tokens: A list of tokens to use for encoding.

    Returns:
        A one-hot encoded list.
    """
    return [int(feature == token) for token in tokens]


def pad_list(l: List[Any], length: int, item: Any) -> List[Any]:  # noqa: E741
    """Pads a list to the specified length.

    Note that if the input list is longer than the specified length it is not
    truncated.

    Args:
        l: The input list.
        length: The desired length of the list.
        item: The item to use to pad the list.

    Returns:
        The padded list.
    """
    return l + [item] * (length - len(l))


def zip_expand(*args: Any):
    """Zips the arguments, expanding them to the largest length if required.

    This will perform a zip of the specified arguments. The resulting list
    will be of the same length as the longest input. If any arguments are
    not lists then they will be repeated to the required length.

    Args:
        args: The elements to zip.

    Returns:
        The zipped list.
    """

    size = max([_len_arg(arg) for arg in args])
    args_list = [_expand_arg(arg, size) for arg in args]
    return zip(*args_list)


def dict_expand(d: Dict[Any, Any]) -> List[Dict[Any, Any]]:
    """Converts a dictionary of lists to a list of dictionaries.

    The resulting list will be of the same length as the longest dictionary
    value. If any values are not lists then they will be repeated to the
    required length.

    Args:
        d: The dictionary of arrays to expand.

    Returns:
        The resulting list of dictionaries.
    """

    size = max([_len_arg(arg) for arg in d.values()])
    d = {k: _expand_arg(v, size) for k, v in d.items()}
    return [{k: v[i] for k, v in d.items()} for i in range(size)]


def _is_iterable(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


def _len_arg(arg: Any) -> int:
    return len(arg) if _is_iterable(arg) else 1


def _expand_arg(arg: Any, size: int) -> List[Any]:
    return arg if _is_iterable(arg) else [arg] * size
