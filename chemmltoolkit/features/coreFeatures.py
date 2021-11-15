from chemmltoolkit.features.utils import get_token_name


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

    _normalize.__name__ = \
        f'normalize({feature.__name__}, mean={mean}, std={std})'
    return _normalize


def one_hot(feature, tokens=None, unknown_token=False):
    """Wraps a feature to one-hot encode the value

    This returns a new feature that first calls the specified feature, then
    returns the result as a one-hot encoding of the specified token list.

    If the 'tokens' parameter is not specified, the default value for the
    feature is infered from a 'tokenizable_feature' decorator if present.

    Args:
        feature: The input feature to one-hot encode.
        tokens: A list of tokens to use for encoding.
        unknown_token: Whether to include an addition token as the final
            index to represent unknown values.

    Returns:
        The one-hot encoded feature.
    """
    if tokens is None:
        tokens = feature.tokens

    feature_name = feature.__name__
    token_names = [get_token_name(token) for token in tokens]

    if unknown_token:
        def _one_hot(input):
            value = feature(input)
            if value in tokens:
                return [int(value == token) for token in tokens] + [0]
            else:
                return [0] * len(tokens) + [1]
    else:
        def _one_hot(input):
            value = feature(input)
            return [int(value == token) for token in tokens]

    def _get_feature_keys():
        features_keys = [f'one_hot({feature_name})[{token_name}]'
                         for token_name in token_names]
        if unknown_token:
            features_keys.append(f'one_hot({feature_name})[?]')
        return features_keys

    token_name_list = ','.join(token_names)
    _one_hot.__name__ = f'one_hot({feature_name}, ' + \
                        f'tokens=[{token_name_list}], ' + \
                        f'unknown_token={unknown_token})'
    _one_hot.get_feature_keys = _get_feature_keys
    return _one_hot
