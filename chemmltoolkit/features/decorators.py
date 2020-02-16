def normalizable_feature(mean, std):
    """Decorator for features to specify default normalization.

    Args:
        mean: The mean value for the feature.
        std: The standard deviation for the feature.
    """
    def _normalizable_feature(func):
        func.normal_mean = mean
        func.normal_std = std
        return func
    return _normalizable_feature


def tokenizable_feature(tokens):
    """Decorator for features to specify default tokens.

    Args:
        tokens: The default set of tokens for the feature.
    """
    def _tokenizable_feature(func):
        func.tokens = tokens
        return func
    return _tokenizable_feature
