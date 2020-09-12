# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.
#
import random
import time
import numpy as np

from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class RandomSearch(HPOptimizationAbstract):

    def __init__(self, **kwargs):
        # inheritance init
        super(RandomSearch, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    ##### implement methods
    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._n_steps = self._n_steps

    ## random search main function
    def _generate(self, param_list, score_list):
        best_param_list = self._population(param_list)

        result_param_list = self._remove_duplicate_params(best_param_list)
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
    ### random search private functions
    def _population(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list