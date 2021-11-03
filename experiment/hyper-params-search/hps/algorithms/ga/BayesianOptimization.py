from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract
from hps.algorithms.HPOptimizerUtil import HPOptimizerUtil
from bayes_opt import BayesianOptimization as bayesian_opt
import numpy as np

class BayesianOptimization(HPOptimizationAbstract):
    def __init__(self, **kwargs):
        super(BayesianOptimization, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    # TODO : change hpo_params.key as Bayesian Opt
    # ----- implement methods
    def _check_hpo_params(self):
        self.init_points = self._hpo_params["init_points"]
        self._n_steps = self._hpo_params["n_steps"]

    # override
    def optimize(self):
        optimizer=bayesian_opt(f=self._learn,
           pbounds=self._pbounds,
           random_state=1,)
        optimizer.maximize(init_points=self.init_points, n_iter=self._n_steps)
        total_result_list = optimizer.res
        return self._make_best_params_bo(total_result_list)


    def _learn(self, **hyper_params):
        curr_job = 0
        temp_hash = HPOptimizerUtil.param_dict_to_hash(hyper_params)
        param_dict = self._make_learning_param_dict_bo(hyper_params)
        self.job_queue.put([temp_hash, param_dict])
        curr_job += 1

        while curr_job > 0:
            results = self.result_queue.get()

            # ---- score
            score = float(results["results"][-1][self._eval_key])

            curr_job -= 1

        return score

    # --- result
    def _make_best_params_bo(self, result_list):
        best_params_list = list()
        sorted_index_list = sorted(result_list, key=(lambda x:x['target']), reverse=True)
        for ins in sorted_index_list[:self.k]:
            best_params_list.append(ins)
        return best_params_list

    def _make_learning_param_dict_bo(self, hyper_params):
        temp_dict = dict()
        model_nm = "-".join([self.hps_info["hpo_alg"], self.hps_info["ml_alg"]])
        temp_dict["model_nm"] = model_nm

        return dict(temp_dict, **self.hps_info["ml_params"]["model_param"], **hyper_params)