import tensorflow as tf
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute import MirroredStrategy
from tensorflow.distribute.experimental import TPUStrategy


def determine_strategy(allow_cpu=False, allow_tpu=True):
    tpu = None
    if allow_tpu:
        try:
            tpu = TPUClusterResolver()
        except ValueError:
            pass

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = TPUStrategy(tpu)
        strategy_type = 'tpu'
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    else:
        gpus = tf.config.experimental.list_logical_devices("GPU")
        if len(gpus) > 1:
            strategy = MirroredStrategy([gpu.name for gpu in gpus])
            strategy_type = 'gpu'
            print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
        elif len(gpus) == 1:
            strategy = tf.distribute.get_strategy()
            strategy_type = 'gpu'
            print('Running on single GPU ', gpus[0].name)
        else:
            if not allow_cpu:
                raise RuntimeError('TensorFlow hardware acceleration ' +
                                   'not found.')

            strategy = tf.distribute.get_strategy()
            strategy_type = 'cpu'
            print('Running on CPU')

    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy, strategy_type
