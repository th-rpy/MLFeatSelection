from attr import attributes
from importlib_metadata import distribution
import pandas as pd
import numpy as np
import glob, os
from tables import Description
from tabulate import tabulate
import pkg_resources
import sys


class Data:
    """ """
    full_path = os.path.realpath(__file__)
    data_paths = glob.glob(os.path.dirname(full_path) + "/OUR_DATA/*.csv")

    def __init__(self) -> None:
        self.data = None
        self.label = None
        self.name = None
        
    def get_dataset_by_name(self, dataset_name):
        
        # function to retrieve the data
        self.name = dataset_name
        stream = pkg_resources.resource_stream(__name__, 'OUR_DATA/' + dataset_name + '.csv')
        df = pd.read_csv(stream, header=None, sep=',')
        self.data = df.iloc[:, 0:-1]
        self.target = df.iloc[:, -1]
        print("Requested dataset found and loaded... \n")
        print("Dataset: " + dataset_name + "\n") 
        print("Shape of data: " + str(self.data.shape) + "\n")
        print("Shape of target: " + str(self.target.shape) + "\n")
        print("==========================================\n") 
        print(df.head(5))
        return self.data, self.target, self.name
    
    def get_dataset_names(self):
        
        list_datasets = [os.path.splitext(os.path.basename(filename))[0] for filename in Data.data_paths]

        return list_datasets
    
    def display_datasets(self):
        # inner function to display the available dataset
        list_datasets = [os.path.splitext(os.path.basename(filename))[0] for filename in Data.data_paths]
        print("\n=========== Available Datasets ===========\n")
        table_list = []

        for i, dataset in enumerate(list_datasets):
            table_list.append([i+1, list_datasets[i]])

        print(tabulate(table_list, headers=["Index", "Dataset"]))    
    
    def get_dataset_by_path(self, dataset_path, target, delimiter=','):
        
        self.name = dataset_path.split('/')[-1].split('.')[0]
        df = pd.read_csv(dataset_path, sep=delimiter, ) # read the dataset
        columns = df.columns # get the columns names
        assert target in columns, "Target column not found in dataset" # check if the target column is in the dataset
        self.data = df.loc[:, df.columns != target] # get the data
        self.target = df[target] # get the target
        
        return self.data, self.target, self.name # return the data and target
    
    def describe_data(self): 
        description_list = []
        name = self.name
        lines, columns = self.data.shape
        attributes = self.data.columns
        n_classes = len(np.unique(self.target))
        distribution_classes = self.target.value_counts()
        description_list.append([name,
                           lines,
                          len(attributes),
                           n_classes,
                           tuple(distribution_classes)])
        print(tabulate(description_list, headers=["Dataset", "Lines", "Attributes", "Classes", "Distribution"]))
        return description_list
        
    def __str__(self):
        return str(self.display_datasets())
        
d = Data()
#d.display_datasets()
d.get_dataset_by_path('mlfeatselection/mlfeatselection/Datasets/iris.csv', target = 'Species', delimiter=',')
d.describe_data()

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
