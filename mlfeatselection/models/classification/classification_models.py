# Loading libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators

classifiers = [est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
cls = [est[1].__name__ for est in classifiers]
print(cls)


class ClassificationModels:

    """Class to define, train and test classification models."""

    def __init__(self):
        self.model = None  # model to be trained
        self.name = None  # name of the model

    def get_list_models(self):

        classifiers = [
            est for est in all_estimators() if issubclass(est[1], ClassifierMixin)
        ]  # get the list of the classifiers
        models_name = [
            est[1].__name__ for est in classifiers
        ]  # get the list of the models names

        return models_name  # return the list of the models names


cd = sklearn.ensemble._weight_boosting.AdaBoostClassifier()
