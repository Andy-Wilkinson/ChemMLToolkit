"""
This type stub file was generated by pyright.
"""

from sklearn.svm import SVR
from sklearn.base import RegressorMixin
from sklearn.neural_network import MLPRegressor

"""Collection of regressors models"""
class OddtRegressor(RegressorMixin):
    _model = ...
    def __init__(self, *args, **kwargs) -> None:
        """ Assemble Neural network or SVM using sklearn pipeline """
        ...
    
    def get_params(self, deep=...): # -> dict[Unknown, Unknown]:
        ...
    
    def set_params(self, **kwargs): # -> Pipeline:
        ...
    
    def fit(self, descs, target_values, **kwargs): # -> OddtRegressor:
        ...
    
    def predict(self, descs):
        ...
    
    def score(self, descs, target_values):
        ...
    


class neuralnetwork(OddtRegressor):
    _model = MLPRegressor
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class svm(OddtRegressor):
    _model = SVR
    def __init__(self, *args, **kwargs) -> None:
        ...
    

