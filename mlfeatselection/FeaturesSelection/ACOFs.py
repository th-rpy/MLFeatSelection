import pandas
import random
import numpy as np
import math
import gc
import copy


class ACOFS:
    def __init__(self, data, target, maxIteration, antNumber, cc, Q, e):
        self.data = data
        self.target = target
        self.fp = [cc] * (len(data.columns))
        self.maxIteration = maxIteration
        self.t_percent = t_percent
        self.ants = []
        self.size = len(data.columns) - 1
        self.antNumber = antNumber
        self.Q = Q
        self.bestScore = 0
        self.result = []
        self.evaporate = e
        self.colonyMax = 0
        self.colonyQuality = 0

    def check_aco_args(self):
        msg_err = ""
        try:
            int(self.t_percent)
        except Exception as e:
            msg_err = "t_percent should be integer!"
            return (False, msg_err)

        try:
            int(self.maxIteration)
        except Exception as e:
            msg_err = "iter_num should be integer!"
            return (False, msg_err)

        if self.maxIteration > 100:
            msg_err = "iter_num should be less than 100!"
            return (False, msg_err)

        if self.maxIteration < 5:
            msg_err = "iter_num should be more than 5!"
            return (False, msg_err)

        return (True, msg_err)
    
