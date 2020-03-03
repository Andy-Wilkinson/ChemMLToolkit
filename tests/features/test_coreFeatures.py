from chemmltoolkit.features.coreFeatures import one_hot
from chemmltoolkit.features.coreFeatures import normalize
from chemmltoolkit.features.decorators import tokenizable_feature
from chemmltoolkit.features.decorators import normalizable_feature
from chemmltoolkit.features.utils import get_feature_keys


class TestCoreFeatures(object):
    def test_one_hot(self):
        feature = one_hot(feature_alpha, tokens=['A', 'B', 'C', 'X', 'Y'])

        assert feature(3) == [0, 1, 0, 0, 0]
        assert feature.__name__ == 'one_hot(feature_alpha, tokens=[A,B,C,X,Y])'
        assert get_feature_keys(feature) == [
                'one_hot(feature_alpha)[A]',
                'one_hot(feature_alpha)[B]',
                'one_hot(feature_alpha)[C]',
                'one_hot(feature_alpha)[X]',
                'one_hot(feature_alpha)[Y]'
            ]

    def test_one_hot_tokenizable(self):
        feature = one_hot(feature_alpha_tokenizable)

        assert feature(3) == [0, 1, 0, 0, 0]
        assert feature.__name__ == \
            'one_hot(feature_alpha_tokenizable, tokens=[A,B,C,X,Y])'
        assert get_feature_keys(feature) == [
                'one_hot(feature_alpha_tokenizable)[A]',
                'one_hot(feature_alpha_tokenizable)[B]',
                'one_hot(feature_alpha_tokenizable)[C]',
                'one_hot(feature_alpha_tokenizable)[X]',
                'one_hot(feature_alpha_tokenizable)[Y]'
            ]

    def test_normalize(self):
        feature = normalize(feature_double, mean=4, std=2)

        assert feature(1) == -1
        assert feature(2) == 0
        assert feature(3) == 1

        assert feature.__name__ == \
            'normalize(feature_double, mean=4, std=2)'

    def test_normalize_normalizable(self):
        feature = normalize(feature_double_normalizable)

        assert feature(1) == -1
        assert feature(2) == 0
        assert feature(3) == 1

        assert feature.__name__ == \
            'normalize(feature_double_normalizable, mean=4, std=2)'


def feature_double(val: int):
    return val * 2


@normalizable_feature(4, 2)
def feature_double_normalizable(val: int):
    return val * 2


def feature_alpha(val: int):
    return ['C', 'A', 'X', 'B', 'Y'][val]


@tokenizable_feature(['A', 'B', 'C', 'X', 'Y'])
def feature_alpha_tokenizable(val: int):
    return ['C', 'A', 'X', 'B', 'Y'][val]
