# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.
from typing import List

import numpy as np


# class : HPOptimizerUtil
class HPOptimizerUtil(object):
    # ----------------------------------------------------------------------------------------------------------------
    # Generate Parameters
    @staticmethod
    def get_param_bound(key):
        return {
            "dropout_prob": [0.0, 0.5],
            "optimizer_fn": ["Adam", "rmsprop", "adadelta"],
            "learning_rate": [0.0, 0.8],
            "act_fn": ["Tanh", "ReLU", "Sigmoid"],
            "hidden_units": [3, 1024],
        }.get(key, None)

    @staticmethod
    def _generate_int_list_params(bound):
        tmp_list = list()
        for i in range(np.random.randint(1, int(bound[0]))):
            tmp_list.append(int(np.random.randint(1, int(bound[1]))))
        return ",".join(map(str, tmp_list))

    @staticmethod
    def _generate_single_params(bound):
        if len(bound) >= 2:
            if type(bound[1]) == int:
                return int(np.random.randint(bound[0], bound[1]))
            elif type(bound[1]) == float:
                return np.random.uniform(bound[0], bound[1])
            else:
                return str(np.random.choice(bound))
        else:
            return str(np.random.choice(bound))

    @staticmethod
    def gen_param(bound, key):
        if key == 'hidden_units' or key == "filter_sizes" or key == "pool_sizes":
            return HPOptimizerUtil._generate_int_list_params(bound)
        else:
            return HPOptimizerUtil._generate_single_params(bound)

    @staticmethod
    def gen_bound_dict(auto_params):
        bound_dict = dict()
        for key in auto_params.keys():
            bound_dict[key] = HPOptimizerUtil.get_param_bound(key)
        return bound_dict

    @staticmethod
    def gen_param_dict_to_bound_dict(bound_dict):
        param_dict = dict()
        for key in bound_dict.keys():
            param_dict[key] = HPOptimizerUtil.gen_param(bound_dict.get(key), key)
        return param_dict

    @staticmethod
    def gen_param_dict_to_auto_params(auto_params):
        return HPOptimizerUtil.gen_param_dict_to_bound_dict(
            HPOptimizerUtil.gen_bound_dict(auto_params)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # DUPLICATE Check
    @staticmethod
    def param_dict_to_hash(param_dict):
        result_str = ""
        for k, v in sorted(dict(param_dict).items()):
            result_str += "{}{}".format(k, v)
        return str(hash(result_str))

    @staticmethod
    def is_duplicate(unique_dict, tmp_hash):
        if unique_dict.get(tmp_hash, None) is None:
            return False
        else:
            return True

    @staticmethod
    def remove_duplicated_params(unique_dict: dict, param_dict_list: list) -> List:
        results = list()
        for i, param_dict in enumerate(param_dict_list):
            temp_hash = HPOptimizerUtil.param_dict_to_hash(param_dict)
            if not HPOptimizerUtil.is_duplicate(unique_dict, temp_hash):
                results.append(param_dict)
        return results

    # ------------------------------------------------------------------------------------------------------------------
    # for Genetic Algorithm
    @staticmethod
    def mutate(bound_dict, param_dict, mut_prob):
        mut_param_dict = dict()
        for key in bound_dict.keys():
            if np.random.rand() > mut_prob:
                mut_param_dict[key] = HPOptimizerUtil.gen_param(bound_dict.get(key), key)
            else:
                mut_param_dict[key] = param_dict[key]

        return mut_param_dict
