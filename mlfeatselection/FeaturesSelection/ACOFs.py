import pandas
import random
import numpy as np
import math
import gc
import copy
import timeit as timeit


class ACOFS:
    def __init__(self, data, target, maxIteration, t_percent, antNumber, cc, Q, e):
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

    def aco_fn(
        self,
        best_individual,
        x_data,
        y_data,
        alpha=1.0,
        beta=0.0,
        t_percent=40,
        iter_num=10,
    ):

        (my_bool, msg_err) = self.check_aco_args()
        if not my_bool:
            print("problem with arguments for ACOFS class!!!")
            print(msg_err)
            exit()  #############

        train_percentage = 100 - int(self.t_percent)

        time_temp = 0
        start = timeit.default_timer()  # ant_num
        (
            best_fitnesses_each_iter,
            average_fitnesses_each_iter,
            num_of_features_selected_by_best_ant_each_iter,
            best_fit_so_far,
            best_ant_road,
        ) = self.run_feature_selection(
            best_individual,
            generations=iter_num,
            alpha=alpha,
            beta=beta,
            T0=0.1,
            T1=0.2,
            Min_T=0.1,
            Max_T=6,
            q=0.95,
            Q=0.3,
            ant_num=50,
            feature_num=len(x_data[1]),
            dataset=x_data,
            targets=y_data,
            train_percentage=train_percentage,
        )
        end = timeit.default_timer()
        time_temp = time_temp + (end - start)

        acc_before_run = self.get_single_fit(x_data, y_data, train_percentage)

        total_feature_num = len(x_data[1])
        sample_num = len(x_data[:, 1])

        best_selected_features_num = np.sum(best_ant_road)
        return (
            best_ant_road,
            acc_before_run,
            best_fit_so_far,
            total_feature_num,
            best_selected_features_num,
            best_fitnesses_each_iter,
            average_fitnesses_each_iter,
            num_of_features_selected_by_best_ant_each_iter,
            time_temp,
            sample_num,
        )

    def ascore(self):
        feature_num = len(self.data.columns) 
        visibility = np.zeros(feature_num * feature_num * 4, dtype="float64").reshape(
            4, feature_num, feature_num
        )

        #     arr = np.corrcoef(dataset)
        #     R = abs(arr)

        print("## F-score :")
        classes = np.unique(self.target)
        class_num = len(classes)
        total_mean_a = self.data.mean(0)
        nominator = 0
        denominator = 0

        #     nominator = np.zeros(feature_num, dtype="int64")
        #     denominator = np.zeros(feature_num, dtype="int64")

        sample_num_of_this_tag = np.zeros(class_num, dtype="int64")
        for i in range(0, class_num):
            tags = np.zeros((len(targets)), dtype="int64")
            bool_arr = np.equal(targets, classes[i])
            tags[bool_arr] = 1
            sample_num_of_this_tag[i] = np.sum(tags)
            dataset_only_class = dataset[bool_arr, :]
            class_mean_a = dataset_only_class.mean(0)
            class_mean_a = np.round(class_mean_a, decimals=4)

            nominator = nominator + np.power(np.subtract(class_mean_a, total_mean_a), 2)
            denominator = denominator + sum(
                np.power(
                    np.subtract(
                        dataset_only_class,
                        np.matlib.repmat(total_mean_a, dataset_only_class.shape[0], 1),
                    ),
                    2,
                )
            ) / (sample_num_of_this_tag[i] - 1)

        Acc_score = np.divide(nominator, denominator)

        visibility[0, :, :] = (0.5 / feature_num) * sum(Acc_score)
        visibility[1, :, :] = np.matlib.repmat(Acc_score, feature_num, 1)

        visibility[2, :, :] = (0.5 / feature_num) * sum(Acc_score)
        visibility[3, :, :] = np.matlib.repmat(Acc_score, feature_num, 1)

        return visibility
