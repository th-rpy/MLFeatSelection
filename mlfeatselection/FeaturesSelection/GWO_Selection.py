import random
import time
import math
import numpy as np
import pandas as pd
import benchmarks
import repackage

repackage.up()
from Datasets.get_data import *


class GwoAlgo:
    """GWO Optimization algo for feature selection."""

    def __init__(self, data, target, objf, lb, ub, SearchAgents_no, Max_iter) -> None:

        self.data = data # dataframe of the dataset to be used for feature selection (data)
        self.target = target # target variable of the dataset to be used for feature selection (target)
        self.objf = objf  # objective function to be optimized
        self.lb = lb  # lower bound of the search space
        self.ub = ub  # upper bound of the search space
        self.dim = self.data.shape[1]  # dimension of the search space
        self.SearchAgents_no = SearchAgents_no  # number of search agents
        self.Max_iter = Max_iter  # maximum number of iterations
        

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

    def optimize(self):
        """Optimize the objective function."""
        (
            Positions,
            Alpha_pos,
            Alpha_score,
            Beta_pos,
            Beta_score,
            Delta_pos,
            Delta_score,
            Convergence_curve,
        ) = (
            self.initiations()
        )  # initialize the variables and get the initialized variables from initiations() function above.

        # Loop counter
        print('GWO is optimizing  "' + self.objf.__name__ + '"')

        # Main loop
        for l in range(0, self.Max_iter):
            for i in range(0, self.SearchAgents_no):

                # Return back the search agents that go beyond the boundaries of the search space
                for j in range(self.dim):
                    Positions[i, j] = np.clip(Positions[i, j], self.lb[j], self.ub[j])

                # Calculate objective function for each search agent
                fitness = self.objf(Positions[i, :])

                # Update Alpha, Beta, and Delta
                if fitness < Alpha_score:
                    Alpha_score = fitness
                    # Update alpha
                    Alpha_pos = Positions[i, :].copy()

                if fitness > Alpha_score and fitness < Beta_score:
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()

                if (
                    fitness > Alpha_score
                    and fitness > Beta_score
                    and fitness < Delta_score
                ):
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()

            a = 2 - l * ((2) / self.Max_iter)
            # a decreases linearly fron 2 to 0

            # Update the Position of search agents including omegas
            for i in range(0, self.SearchAgents_no):
                for j in range(0, self.dim):

                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]

                    A1 = 2 * a * r1 - a
                    # Equation (3.3)
                    C1 = 2 * r2
                    # Equation (3.4)

                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                    # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    # Equation (3.6)-part 1

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a
                    # Equation (3.3)
                    C2 = 2 * r2
                    # Equation (3.4)

                    D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                    # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta
                    # Equation (3.6)-part 2

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    # Equation (3.3)
                    C3 = 2 * r2
                    # Equation (3.4)

                    D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                    # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta
                    # Equation (3.5)-part 3

                    Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
            Convergence_curve[l] = Alpha_score

            if l % 100 == 0:
                print(
                    [
                        "At iteration "
                        + str(l)
                        + " the best fitness is "
                        + str(Alpha_score)
                    ]
                )

        print("Alpha position=", Alpha_pos)
        print("Beta position=", Beta_pos)
        print("Delta position=", Delta_pos)
        return Alpha_pos, Beta_pos

    def select_features(self, threshold=-0.1):
        """Select features for the model."""
        # Select features
        ##Applying feature selection on the given dataset
        ##considering alpha as best solution and putting a threshold
        alpha, _ = self.optimize()
        print("Number of selected features = ", alpha.shape[0])
        selected_data = pd.DataFrame()
        assert (
            data.shape[1] == self.dim
        ), "ERROR ..."  # check if the number of features is correct

        for i in range(0, alpha.shape[0]):
            if alpha[i] >= threshold:
                selected_data[self.data.columns[i]] = self.data[self.data.columns[i]]
        print("The modified data is following")
        # selected_data.head()  # only 491 are selected
        returned_data = Data()
        returned_data.data = selected_data # assign the selected data to the returned_data
        returned_data.target = self.target # assign the target to the returned_data
        return returned_data # return the selected data


if __name__ == "__main__":
    ##setting GWO parameters
    Max_iter = 80
    SearchAgents_no = 35
    dimension = 503
    search_domain = [0, 1]
    lb = -1.0
    ub = 1.0
    d = Data()
    # d.display_datasets()
    d.get_dataset_by_path(
        "mlfeatselection/mlfeatselection/Datasets/iris.csv", target="Species", delimiter=","
    )
    
    data, target = d.data_preprocessing(vars_to_drop= ['Id'])

    func_details = benchmarks.getFunctionDetails(6)
    f = getattr(benchmarks, "F5")
    g = GwoAlgo(data, target, f, lb, ub, SearchAgents_no, Max_iter)
    for i in range(0, 1):
        alpha, beta = g.optimize()
    
    D = g.select_features(threshold=0.)
    
    print("The modified data is following \n") 
    print(D.data.head())


