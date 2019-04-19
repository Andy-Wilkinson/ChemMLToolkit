from tensorflow.keras.utils import get_custom_objects


def register_keras_custom_object(cls):
    get_custom_objects()[cls.__name__] = cls
    return cls
