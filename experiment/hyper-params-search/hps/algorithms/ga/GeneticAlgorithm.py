# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 
#
import random
import time
import numpy as np

from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class GeneticAlgorithm(HPOptimizationAbstract):

    def __init__(self, **kwargs):
        # inheritance init
        super(GeneticAlgorithm, self).__init__(**kwargs)
        self._check_hpo_params()

    ##### implement methods
    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._mut_prob = self._hpo_params["mut_prob"]
        self._cx_prob = self._hpo_params["cx_prob"]
        self._n_sel = int(float(self._hpo_params["sel_ratio"] * self._n_pop))
        self._n_mut = int(float(self._hpo_params["mut_ratio"] * self._n_pop))
        self._n_cx = int(float(self._hpo_params["cx_ratio"] * self._n_pop))
        self._n_steps = self._hpo_params["n_steps"]

    ## genetic algorithm main function
    def _generate(self, param_list, score_list):
        result_param_list = list()
        best_param_list = self._population(param_list)
        sel_params = self._selection(best_param_list)
        mut_params = self._mutation(best_param_list)
        cx_params = self._crossover(best_param_list)

        result_param_list += sel_params + mut_params + cx_params
        result_param_list = self._remove_duplicate_params(result_param_list)
        num_result_params = len(result_param_list)

        ## leak
        if  num_result_params < self._n_pop:
            result_param_list += self._generate_param_dict_list(self._n_pop - num_result_params)
        ## over
        elif num_result_params > self._n_pop :
            random.shuffle(result_param_list)
            result_param_list = result_param_list[:self._n_pop]
        return result_param_list

    #############################################################################
    ### Genetic Algorithm private functions
    def _population(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list

    def _selection(self, param_dict_list):
        return param_dict_list[:self._n_sel]

    def _mutation(self, param_dict_list):
        mut_params = list()
        for param_dict in param_dict_list[:self._n_mut]:
            temp_param_dict = dict()
            for _ , key in enumerate(self._pbounds):
                if np.random.rand() > self._mut_prob:
                    temp_param_dict[key] = self._generate_param(key)
                else :
                    temp_param_dict[key] = param_dict[key]
            mut_params.append(temp_param_dict)
        return mut_params

    def _crossover(self, params):
        cx_params = list()
        for i in range(int(self._n_cx/2)):
            sampled_param_dict_list = random.sample(params, 2)

            min_len = len(sampled_param_dict_list[0])
            threshold = np.random.randint(0, min_len)

            temp_dict_first = dict()
            temp_dict_second = dict()
            for idx, key in enumerate(sampled_param_dict_list[0].keys()):
                ### origin
                if idx > threshold:
                    temp_dict_first[key] = sampled_param_dict_list[0][key]
                    temp_dict_second[key] = sampled_param_dict_list[1][key]
                ### crossover
                else:
                    temp_dict_first[key] = sampled_param_dict_list[1][key]
                    temp_dict_second[key] = sampled_param_dict_list[0][key]

            cx_params.append(temp_dict_first)
            cx_params.append(temp_dict_second)

        return cx_params



if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
                "mut_prob" : 0.5,
                "cx_prob" : 0.5,
                "sel_ratio" : 0.5,
                "mut_ratio" : 0.25,
                "cx_ratio" : 0.25,
                "n_steps" : 10,
                "n_params" : 10,
                "k_val" : 5,
                "eval_key" : "accuracy"
            },
        "ml_params":{
            "model_param":{
                "input_units" : "100",
                "output_units" : "1",
                "global_step" : "10",
                "early_type" : "2",
                "min_step" : "10",
                "early_key" : "accuracy",
                "early_value" : "0.98",
                "method_type" : "Basic",
                "global_sn" : "0",
                "alg_sn" : "0",
                "algorithm_type" : "classifier",
                "job_type" : "learn"
            },
            "pbounds":{
                "dropout_prob": [0, 0.5],
                "optimizer_fn": ["Adam", "rmsprop", "Adadelta"],
                "learning_rate": [0, 0.8],
                "act_fn": ["Tanh", "ReLU", "Sigmoid"],
                "hidden_units" : [3,1024]
            }
        }
    }
    ga = GeneticAlgorithm(hprs_info=hprs_info)
    best_params = ga._generate([], [])
    print(best_params)
