from typing import Any, List
from chemmltoolkit.utils.list_utils import flatten
from chemmltoolkit.utils.list_utils import _is_iterable
from chemmltoolkit.features.utils import get_feature_keys


class Featuriser:
    """A generic feature generator.

    This will generally be the base class for specific featurisers.

    Args:
        features: A list of features to generate.
    """
    def __init__(self, features: list):
        self.features = features

    def _process(self, data) -> List[Any]:
        """Generates features for an individual data point.

        Args:
            data: The data to featurise.

        Returns:
            A list of features.
        """

        features = [feature(data) for feature in self.features]
        return flatten(features)

    def _get_feature_lengths(self, example) -> List[int]:
        """Calculates the length of each feature

        This uses an example data item, generate each feature in turn
        and return a list of each length.

        Args:
            data: Example data to featurise.

        Returns:
            A list of the lengths of each feature.
        """
        def _get_feature_length(feature):
            val = feature(example)
            if _is_iterable(val):
                return len(val)
            else:
                return 1

        return [_get_feature_length(feature) for feature in self.features]

    def get_feature_names(self) -> List[str]:
        """Gets strings representing each type of feature

        This will return a list of strings, one for each feature function
        specified when creating the featuriser. This can be used for example
        to log the features chosen, or to save them for recreation later.

        Returns:
            A list of feature descriptions.
        """
        return flatten([feature.__name__ for feature in self.features])

    def get_feature_keys(self) -> List[str]:
        """Gets the name of each element in the generated feature

        This will return a list of strings, one for each output value when
        generating features. You can use this to display a human-readable
        description of each element of the feature (e.g. when understanding
        feature importance). This will typically be the name of the feature,
        with some form of index appended if the feature outputs multiple
        values.

        Returns:
            A list of feature names.
        """
        lengths = self.get_feature_lengths()

        return flatten([get_feature_keys(feature, length)
                        for feature, length
                        in zip(self.features, lengths)])
