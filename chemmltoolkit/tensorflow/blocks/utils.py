from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


def get_batchnorm(batchnorm):
    if not batchnorm:
        return None
    if batchnorm == 'norm':
        return BatchNormalization(renorm=False)
    if batchnorm == 'renorm':
        return BatchNormalization(renorm=True)
    raise ValueError(f'Invalid value `{batchnorm}` for `batchnorm` parameter')


def get_dropout(dropout):
    if dropout > 0.0:
        return Dropout(dropout)
    else:
        return None
