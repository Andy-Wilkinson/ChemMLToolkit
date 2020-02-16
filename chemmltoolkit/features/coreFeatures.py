from chemmltoolkit.utils import list_utils


def normalize(feature, mean=None, std=None):
    """Wraps a feature to normalize the value

    This returns a new feature that first calls the specified feature, then
    returns the result normalized by the transformation '(value - mean) / std'.

    If the 'mean' or 'std' parameters are not specified, the default values
    for the feature are infered from a 'normalizable_feature' decorator if
    present.

    Args:
        feature: The input feature to normalize.
        mean: The mean value for the feature.
        std: The standard deviation for the feature.

    Returns:
        The normalized feature.
    """
    if mean is None:
        mean = feature.normal_mean
    if std is None:
        std = feature.normal_std

    def _normalize(input):
        return (feature(input) - mean) / std
    return _normalize


def one_hot(feature, tokens=None):
    """Wraps a feature to one-hot encode the value

    This returns a new feature that first calls the specified feature, then
    returns the result as a one-hot encoding of the specified token list.

    If the 'tokens' parameter is not specified, the default value for the
    feature is infered from a 'tokenizable_feature' decorator if present.

    Args:
        feature: The input feature to one-hot encode.
        tokens: A list of tokens to use for encoding.

    Returns:
        The normalized feature.
    """
    if tokens is None:
        tokens = feature.tokens

    def _one_hot(input):
        return list_utils.one_hot(feature(input), tokens)
    return _one_hot
