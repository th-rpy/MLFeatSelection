import pandas as pd
import numpy as np
import glob, os
from tabulate import tabulate
import pkg_resources
import sys


class Data:
    """ """

    def __init__(self) -> None:
        self.data = None
        self.label = None
        
    def get_dataset(self, dataset_name):
        
        # function to retrieve the data
        full_path = os.path.realpath(__file__)
        data_paths = glob.glob(os.path.dirname(full_path) + "/OUR_DATA/*.csv")

        stream = pkg_resources.resource_stream(__name__, 'OUR_DATA/' + dataset_name + '.csv')
        df = pd.read_csv(stream, header=1, sep=',')
        self.data = np.array(df.iloc[:, 0:-1])
        self.target = np.array(df.iloc[:, -1])
        print("Requested dataset found and loaded...")

        return self.data, self.target 
    
    def get_dataset_names(self):
        # function to retrieve the data
        full_path = os.path.realpath(__file__)
        data_paths = glob.glob(os.path.dirname(full_path) + "/OUR_DATA/*.csv")
        list_datasets = [os.path.splitext(os.path.basename(filename))[0] for filename in data_paths]

        return list_datasets
    
    def __str__(self):
        return str(self.get_dataset_names())
    
    
d = Data()
d.get_dataset('iris')
print(d)

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
