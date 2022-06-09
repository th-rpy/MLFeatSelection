# Loading libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators

classifiers = [est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
cls = [est[1].__name__ for est in classifiers]
print({est[1].__name__: est[1] for est in classifiers}["LogisticRegression"])


class ClassificationModels:

    """Class to define, train and test classification models."""

    classifiers = [
        est for est in all_estimators() if issubclass(est[1], ClassifierMixin)
    ]
    models_dict = {est[1].__name__: est[1] for est in classifiers}

    def __init__(self, model=None):
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

    def define_model_by_name(self, model_name):
        """Define the model by its name"""
        assert (
            model_name in self.get_list_models()
        ), "Model not found, please check the name of the model. There are our models {}".format(
            self.get_list_models()
        )
        function_string = ClassificationModels.models_dict[
            model_name
        ]  # get the function string
        function_string = str(function_string).split(" ")[1][
            1:-2
        ]  # get the function string
        mod_name, func_name = function_string.rsplit(
            ".", 1
        )  # get the module name and the function name
        mod = importlib.import_module(mod_name)  # import the module
        func = getattr(mod, func_name)  # get the function
        self.model = func()  # define the model
        return ClassificationModels(self.model)  # return the model

    def fit_model(self, X, y):
        """Fit the model"""
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"

        try:
            self.model.fit(X, y)
        except Exception as e:
            print(e)
            print("Model not trained")

    def predict_model(self, X):
        """Predict the model"""
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        try:
            y_predicted = self.model.predict(X)
        except Exception as e:
            print(e)
            print("Model not predicted")
            y_predicted = np.zeros(X.shape[0])

        return y_predicted

    def get_model_accuracy(self, y_test, y_pred):
        """Get the model accuracy"""
        assert isinstance(y_test, np.ndarray), "y_test must be a numpy array"
        assert isinstance(y_pred, np.ndarray), "y_pred must be a numpy array"
        assert (
            y_test.shape == y_pred.shape
        ), "y_test and y_pred must have the same shape"
        try:
            accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        except Exception as e:
            print(e)
            print("Model accuracy not calculated")
            accuracy = 0
        return accuracy

    def get_model_confusion_matrix(self, y_test, y_pred):
        """Get the model confusion matrix"""
        assert isinstance(y_test, np.ndarray), "y_test must be a numpy array"
        assert isinstance(y_pred, np.ndarray), "y_pred must be a numpy array"
        assert (
            y_test.shape == y_pred.shape
        ), "y_test and y_pred must have the same shape"
        try:
            cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        except Exception as e:
            print(e)
            print("Model confusion matrix not calculated")
            cm = np.zeros((2, 2))
        return cm

    def display_model_confusion_matrix(self, X_test, y_test):
        """Display the model confusion matrix"""
        assert isinstance(y_test, np.ndarray), "y_test must be a numpy array"
        assert isinstance(X_test, np.ndarray), "X_test must be a numpy array"

        try:
            display = ConfusionMatrixDisplay.from_estimator(
                self.model,
                X_test,
                y_test.reshape(-1, 1),
            )
            display.ax_.set_title("Model confusion matrix")
            print(display.confusion_matrix)
            plt.show()
        except Exception as e:
            print(e)
            print("Model confusion matrix not calculated")
            cm = np.zeros((2, 2))


clsModel = ClassificationModels()
LogisticCls = clsModel.define_model_by_name("LogisticRegression")
clsModel.fit_model(
    np.array([[1, 2, 3], [4, 8, 9], [10, -1, 63]]), np.array([0, 1, 0]).reshape(-1, 1)
)
y_pred = clsModel.predict_model(np.array([[1, 2, 2.13]]))
print(y_pred)
disp = ConfusionMatrixDisplay.from_estimator(
    clsModel.model,
    np.array([[1, 2, 2.13], [4, 8, 9], [10, -1, 63]]),
    np.array([0, 1, 0]).reshape(-1, 1),
)
disp.ax_.set_title("title")
print(disp.confusion_matrix)

plt.show()
"""import importlib


function_string = str(classifiers[0][1]).split(" ")[1][1:-2]
mod_name, func_name = function_string.rsplit(".", 1)
mod = importlib.import_module(mod_name)
func = getattr(mod, func_name)
result = func()
print(type(result))
print(str(classifiers[0][1]).split(' ')[1][1:-2] == function_string)"""
