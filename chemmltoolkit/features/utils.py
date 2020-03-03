def get_feature_keys(feature, feature_length=1):
    if hasattr(feature, 'get_feature_keys'):
        return feature.get_feature_keys()
    else:
        feature_name = feature.__name__
        if feature_length > 1:
            return [f'{feature_name}[{i}]' for i in range(feature_length)]
        else:
            return [feature_name]


def get_token_name(token):
    return str(token)
