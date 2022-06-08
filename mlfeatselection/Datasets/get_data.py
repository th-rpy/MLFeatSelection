from attr import attributes
from importlib_metadata import distribution
import pandas as pd
import numpy as np
import glob, os
from tables import Description
from tabulate import tabulate
import pkg_resources
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer


class Data:

    """Class to get the data from the dataset, store it in a class and preprocessing it."""

    full_path = os.path.realpath(__file__) # get the full path of the file
    data_paths = glob.glob(os.path.dirname(full_path) + "/OUR_DATA/*.csv") # get the data paths

    def __init__(self) -> None:
        self.data = None  # data
        self.label = None  # label
        self.name = None  # name

    def get_dataset_by_name(self, dataset_name):

        # function to retrieve the data
        self.name = dataset_name
        stream = pkg_resources.resource_stream(
            __name__, "OUR_DATA/" + dataset_name + ".csv"
        ) # get the stream of the dataset
        df = pd.read_csv(stream, header=None, sep=",") # read the dataset
        self.data = df.iloc[:, 0:-1] # get the data
        self.target = df.iloc[:, -1] # get the target
        print("Requested dataset found and loaded... \n")
        print("Dataset: " + dataset_name + "\n")
        print("Shape of data: " + str(self.data.shape) + "\n")
        print("Shape of target: " + str(self.target.shape) + "\n")
        print("==========================================\n")
        print(df.head(5))
        return self.data, self.target, self.name # return the data, target and name

    def get_dataset_names(self):

        list_datasets = [
            os.path.splitext(os.path.basename(filename))[0]
            for filename in Data.data_paths
        ] # get the list of the datasets names from the data paths

        return list_datasets # return the list of the datasets names

    def display_datasets(self):
        # inner function to display the available dataset
        list_datasets = [
            os.path.splitext(os.path.basename(filename))[0]
            for filename in Data.data_paths
        ]
        print("\n=========== Available Datasets ===========\n")
        table_list = []

        for i, dataset in enumerate(list_datasets):
            table_list.append([i + 1, list_datasets[i]]) # append the dataset to the list

        print(tabulate(table_list, headers=["Index", "Dataset"])) # print the table

    def get_dataset_by_path(self, dataset_path, target, delimiter=","):

        self.name = dataset_path.split("/")[-1].split(".")[0]
        df = pd.read_csv(
            dataset_path,
            sep=delimiter,
        )  # read the dataset
        columns = df.columns  # get the columns names
        assert (
            target in columns
        ), "Target column not found in dataset"  # check if the target column is in the dataset
        self.data = df.loc[:, df.columns != target]  # get the data
        self.target = df[target]  # get the target

        return self.data, self.target, self.name  # return the data and target

    def describe_data(self):

        description_list = []  # list to store the description of the data
        name = self.name  # get the name of the dataset
        lines, columns = self.data.shape  # get the shape of the data
        attributes = self.data.columns  # get the attributes of the data
        n_classes = len(np.unique(self.target))  # get the number of classes
        distribution_classes = (
            self.target.value_counts()
        )  # get the distribution of the classes
        description_list.append(
            [name, lines, len(attributes), n_classes, tuple(distribution_classes)]
        )  # append the description to the list
        print("\n=========== Dataset Summary ===========\n")
        print(
            tabulate(
                description_list,
                headers=["Dataset", "Lines", "Attributes", "Classes", "Distribution"],
            )
        )  # print the table
        print("==========================================\n")
        return description_list  # return the description list

    def data_preprocessing(
        self, continuous_columns=None, categorical_columns=None, vars_to_drop=None
    ):

        if vars_to_drop is not None:  # if the user wants to drop some variables
            assert isinstance(
                vars_to_drop, list
            ), "vars_to_drop must be a list"  # check if the vars_to_drop is a list
            for var in vars_to_drop:
                assert var in self.data.columns, "Variable to drop not found in dataset"
                self.data.drop(var, axis=1, inplace=True)

        self.data = self.data.dropna()  # drop the NaN values
        self.data = self.data.drop_duplicates()  # drop the duplicated values

        # Starting preprocessing
        if (
            continuous_columns is None and categorical_columns is None
        ):  # if the user doesn't specify the columns
            num_data = [
                cname
                for cname in self.data.columns
                if self.data[cname].dtypes in ["int64", "float64"]
            ]  # get the numeric columns
            cat_data = list(
                filter(lambda x: x not in num_data, self.data.columns)
            )  # get the categorical data

        elif (
            continuous_columns is not None and categorical_columns is None
        ):  # if the user specifies the continuous columns
            num_data = continuous_columns  # get the numeric columns

        elif (
            continuous_columns is None and categorical_columns is not None
        ):  # if the user specifies the categorical columns
            cat_data = categorical_columns

        else:  # if the user specifies both the continuous and categorical columns
            num_data = continuous_columns  # get the numeric columns
            cat_data = categorical_columns

        ## Normalization of the data ##
        if len(num_data) > 0:  # if there are numeric columns
            for col in num_data:  # for each numeric column
                self.data[col] = self.data[col].astype(
                    "float64"
                )  # convert the column to float64

                self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[
                    col
                ].std()  # normalize the column using z-score normalization (mean = 0, std = 1)

        ## One-Hot Encoding of the categorical data ##
        if len(cat_data) > 0:  # if there are categorical columns
            for col in cat_data:  # for each categorical column
                self.data[col] = pd.get_dummies(
                    self.data[col], prefix=str(col)
                )  # label binarize the column

        return self.data  # return the data

    def __str__(self):
        return str(self.display_datasets())


d = Data()
# d.display_datasets()
d.get_dataset_by_path(
    "mlfeatselection/mlfeatselection/Datasets/iris.csv", target="Species", delimiter=","
)
d.data_preprocessing(vars_to_drop=["Id"])

"""data_paths = glob.glob(os.getcwd() + "/OUR_DATA/*.csv")
list_datasets = [os.path.splitext(os.path.basename(filename))[0] for filename in data_paths]
import os

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(os.path.dirname(full_path))"""
