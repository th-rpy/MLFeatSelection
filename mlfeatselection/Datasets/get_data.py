import pandas as pd
import numpy as np
import glob, os
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
        
    def get_dataset_by_name(self, dataset_name):
        
        # function to retrieve the data
        stream = pkg_resources.resource_stream(__name__, 'OUR_DATA/' + dataset_name + '.csv')
        df = pd.read_csv(stream, header=None, sep=',')
        self.data = np.array(df.iloc[:, 0:-1])
        self.target = np.array(df.iloc[:, -1])
        print("Requested dataset found and loaded... \n")
        print("Dataset: " + dataset_name + "\n") 
        print("Shape of data: " + str(self.data.shape) + "\n")
        print("Shape of target: " + str(self.target.shape) + "\n")
        print("==========================================\n") 
        print(df.head(5))
        return self.data, self.target 
    
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
        
        df = pd.read_csv(dataset_path, sep=delimiter, ) # read the dataset
        columns = df.columns # get the columns names
        assert target in columns, "Target column not found in dataset" # check if the target column is in the dataset
        self.data = np.array(df.loc[:, df.columns != target]) # get the data
        self.target = np.array(df[target]) # get the target
        
        return self.data, self.target # return the data and target
    
    def __str__(self):
        return str(self.display_datasets())
        
d = Data()
#d.display_datasets()
d.get_dataset_by_path('mlfeatselection/mlfeatselection/Datasets/iris.csv', target = 'Species', delimiter=',')
print(d.data)

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
