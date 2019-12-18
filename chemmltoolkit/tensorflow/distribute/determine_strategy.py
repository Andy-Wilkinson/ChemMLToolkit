import tensorflow as tf


def determine_strategy(allow_cpu=False, allow_tpu=True):
    tpu = None
    if allow_tpu:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            pass

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        strategy_type = 'tpu'
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    else:
        gpus = tf.config.experimental.list_logical_devices("GPU")
        if len(gpus) > 1:
            gpu_names = [gpu.name for gpu in gpus]
            strategy = tf.distribute.MirroredStrategy(gpu_names)
            strategy_type = 'gpu'
            print('Running on multiple GPUs ', gpu_names)
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
