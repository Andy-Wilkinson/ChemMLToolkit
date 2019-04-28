from chemmltoolkit.tensorflow.distribute import NoDistributionStrategy


class TestNoDistributionStrategy():
    def test_scope(self):
        strategy = NoDistributionStrategy()
        with strategy.scope():
            assert 1 == 1
