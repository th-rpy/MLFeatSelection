import random
import time
import math
import numpy as np
import pandas as pd
from mlfeatselection.utils import utils


class GwoAlgo:
    """GWO Optimization algo for feature selection."""

    def __init__(self, objf, lb, ub, dim, SearchAgents_no, Max_iter) -> None:

        self.objf = objf  # objective function to be optimized
        self.lb = lb  # lower bound of the search space
        self.ub = ub  # upper bound of the search space
        self.dim = dim  # dimension of the search space
        self.SearchAgents_no = SearchAgents_no  # number of search agents
        self.max_iter = Max_iter  # maximum number of iterations

    def initiations(self):

        # initialize alpha, beta, and delta_pos
        Alpha_pos = np.zeros(self.dim)  # Alpha position
        Alpha_score = float("inf")  # Alpha score

        Beta_pos = np.zeros(self.dim)  # Beta position
        Beta_score = float("inf")  # Beta score

        Delta_pos = np.zeros(self.dim)  # Delta position
        Delta_score = float("inf")  # Delta score

        if not isinstance(self.lb, list):  # if lb is not a list
            self.lb = [self.lb] * self.dim  # make lb a list of the same size as dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim  # make ub a list of the same size as dim

        # Initialize the positions of search agents
        Positions = np.zeros(
            (self.SearchAgents_no, self.dim)
        )  # positions of search agents
        for i in range(self.dim):
            Positions[:, i] = (
                np.random.uniform(0, 1, self.SearchAgents_no)
                * (self.ub[i] - self.lb[i])
                + self.lb[i]
            )  # initialize the positions of search agents

        Convergence_curve = np.zeros(self.Max_iter)  # initialize the convergence curve

        return (
            Positions,
            Alpha_pos,
            Alpha_score,
            Beta_pos,
            Beta_score,
            Delta_pos,
            Delta_score,
            Convergence_curve,
        )  # return the initialized variables
        
        def 
