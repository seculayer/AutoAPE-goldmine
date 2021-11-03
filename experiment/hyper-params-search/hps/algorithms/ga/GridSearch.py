# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.
#
import random
import time
import numpy as np
from hps.common.Common import Common
from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract
from hps.algorithms.HPOptimizerUtil import HPOptimizerUtil

class GridSearch(HPOptimizationAbstract):

    def __init__(self, **kwargs):
        # inheritance init
        super(GridSearch, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False
        self._categorical_param_dict = dict()

        self.categorical_boundary()
        self.LOGGER = Common.LOGGER.getLogger()

    ##### implement methods
    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._n_steps = self._n_steps

    ## grid search main function

    def _generate(self, param_list, score_list):

        best_param_list = self._population(param_list)
        result_param_list = HPOptimizerUtil.remove_duplicated_params(self.unique_param_dict, best_param_list)
        num_result_params = len(result_param_list)

        ## leak
        if  num_result_params < self._n_pop:
            result_param_list += self._generate_categorical_param_dict_list(self._n_pop - num_result_params)
        ## over
        elif num_result_params > self._n_pop :
            random.shuffle(result_param_list)
            result_param_list = result_param_list[:self._n_pop]
        return result_param_list

    #############################################################################
    ### grid search private functions
    def _population(self, param_list):
        if len(param_list) == 0:
            return self._generate_categorical_param_dict_list(self._n_pop)
        else :
            return param_list

    def _generate_categorical_param_dict_list(self, n_pop):
        temp_list = list()
        for i in range(n_pop):
            temp_dict = self._generate_catecorical_param_dict()
            temp_list.append(temp_dict)
        return temp_list

    def categorical_boundary(self):
        for k in self._pbounds.keys():
            if k == "optimizer_fn" or k == "act_fn":
                self._categorical_param_dict[k] = self._pbounds[k]
            elif k=="hidden_units":
                hidden_list=list()
                for i in range(np.random.randint(1, self._pbounds[k][0])):
                    hidden_list.append(int(np.random.randint(1, self._pbounds[k][1])))
                self._categorical_param_dict[k] = hidden_list
            else:
                temp_list = list()
                for i in range (5):
                    temp_list.append(np.random.uniform(self._pbounds[k][0], self._pbounds[k][1]))
                self._categorical_param_dict[k] = temp_list

    def _generate_catecorical_param_dict(self):
        param_dict = dict()
        for k in self._categorical_param_dict.keys():
            param_dict[k] = np.random.choice(self._categorical_param_dict[k])

        tmp_hash = HPOptimizerUtil.param_dict_to_hash(param_dict)
        if not HPOptimizerUtil.is_duplicate(self.unique_param_dict, tmp_hash):
            self.unique_param_dict[tmp_hash] = {"param_dict": param_dict}
            return param_dict
        else:
            return self._generate_catecorical_param_dict()