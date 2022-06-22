import pandas
import random
import numpy as np
import math
import gc
import copy

class ACO:

    def __init__(self,data,maxIteration,antNumber,cc,Q,e):
        self.data = data
        self.fp = [cc]*(len(data.columns)-1)
        self.maxIteration = maxIteration
        self.ants = []
        self.size = len(data.columns)-1
        self.antNumber= antNumber
        self.Q = Q
        self.bestScore = 0
        self.result=[]
        self.evaporate = e
        self.colonyMax = 0
        self.colonyQuality = 0 