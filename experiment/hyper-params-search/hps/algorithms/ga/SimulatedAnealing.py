import random
import time
import numpy as np
from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract


class SimulatedAnnealing(HPOptimizationAbstract):

    def __init__(self, **kwargs):
        # inheritance init
        super(SimulatedAnnealing, self).__init__(**kwargs)
        self._check_hpo_params()

    def _check_hpo_params(self):
        self._n_pop = self._n_params
        self._T0 = self._hpo_params["T0"]
        self._alpha = self._hpo_params["alpha"]
        self._k = self._hpo_params["k"]
        self._n_steps = self._hpo_params["n_steps"]

    def _generate(self, param_list, score_list):
        result_param_list = list()

        # random init population
        best_param_list = self._population(param_list)
        # population 값과 비교할 후보군
        neighbor_selection = self._neighbor(best_param_list)
        # population & 후보군 중 선택하는 방법
        accept_criteria = self._accept(neighbor_selection, best_param_list)

        result_param_list = accept_criteria
        result_param_list = self._remove_duplicate_params(result_param_list)
        return result_param_list


    # population 초기화
    def _population(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list

    # neighbor selection로 candidate 생성
    def _neighbor(self, param_dict_list):
        return self._generate_param_dict_list(self._n_params)

    # accept criteria
    def _accept(self, param_dict_list, best_params):
        best_params_list = list()

        _, of_final = self._learn(self._n_steps, param_dict_list)
        _, of_new = self._learn(self._n_steps, best_params)


        # best값과 neighbor값의 비교
        if of_new <= of_final :
            best_params_list = param_dict_list
        else :
            random_value = np.random.rand()
            form = 1 / (np.exp((of_new[1] - of_final[1]) / self._T0))
            if random_value <= form :
                best_params_list = best_params
            else :
                best_params_list = param_dict_list

        # temperature 조절
        self._T0 = self._alpha * self._T0

        return best_params_list

if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
            "T0" : 0.40,
            "alpha" : 0.85,
            "n_pop" : 1,
            "k" : 0.1,
            "n_params": 10,
            "k_val": 1,
            "eval_key": "accuracy"
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
                "optimizer_fn": "Adam",
                "learning_rate": 0.8,
                "act_fn": "Sigmoid",
                "hidden_units" : 50
            }
        }
    }
    sa = SimulatedAnnealing(hps_info = hprs_info)
    best_params = sa._generate([], [])
    print(best_params)