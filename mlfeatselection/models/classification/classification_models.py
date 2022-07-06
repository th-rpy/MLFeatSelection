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
        self.name = model_name  # define the name of the model
        return ClassificationModels(self.model)  # return the model

    def split_data(self, X, y, test_size=0.2):
        """Split the data in train and test sets"""
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        assert test_size >= 0 and test_size <= 1, "test_size must be between 0 and 1"

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size
        )
        return X_train, X_test, y_train, y_test

    def fit_model(self, X, y):
        """Fit the model"""
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"

        try:
            self.model.fit(X, y.reshape(-1, 1))
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

    def get_model_classification_report(self, y_test, y_pred):
        """Get the model classification report"""
        assert isinstance(y_test, np.ndarray), "y_test must be a numpy array"
        assert isinstance(y_pred, np.ndarray), "y_pred must be a numpy array"
        assert (
            y_test.shape == y_pred.shape
        ), "y_test and y_pred must have the same shape"
        try:
            report = sklearn.metrics.classification_report(y_test, y_pred)
        except Exception as e:
            print(e)
            print("Model classification report not calculated")
            report = "Model classification report not calculated"
        print(report)
        return report

    def get_results_details(self, X_test, y_test):
        y_pred = self.predict_model(X_test)
        d = {
            "index": 0,
            "Model": self.name,
            "Accuracy": sklearn.metrics.accuracy_score(y_test, y_pred),
            "Precision": sklearn.metrics.precision_score(y_test, y_pred),
            "Recall": sklearn.metrics.recall_score(y_test, y_pred),
            "F1 Score": sklearn.metrics.f1_score(y_test, y_pred),
        }
        return d

    def __str__(self):
        return "\n".join(self.get_list_models())


clsModel = ClassificationModels()  # create the model object
LogisticCls = clsModel.define_model_by_name(
    "LogisticRegression"
)  # define the model by name "LogisticRegression"
X = np.array([[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]])  # define the X data
X_test = np.array([[1, 2], [1, 1], [4, 5]])  # define the X test data
y = np.array([0, 0, 1, 1, 0])  # define the y data
y_test = np.array([0, 1, 1])  # define the y test data

# fit the model
clsModel.fit_model(X, y.reshape(-1, 1))
y_pred = clsModel.predict_model(X_test)  # predict the model
print(y_pred)  # print the predicted data
accuracy = clsModel.get_model_accuracy(y_test, y_pred)  # get the model accuracy
clsModel.display_model_confusion_matrix(
    X_test, y_test
)  # display the model confusion matrix
clsModel.get_model_classification_report(
    y_test, y_pred
)  # get the model classification report
print(clsModel.get_results_details(X_test, y_test))  # get the results details
