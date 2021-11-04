import multiprocessing
from typing import List

import numpy as np

from hps.algorithms.HPOptimizerUtil import HPOptimizerUtil
from hps.common.Common import Common


# class : HPOptimizationAbstract
class HPOptimizationAbstract(object):
    def __init__(self, hps_info, job_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue, **kwargs):
        # --- framework objects
        self.LOGGER = Common.LOGGER.getLogger()

        # --- basic variables
        self.hps_info = hps_info
        self.job_queue = job_queue
        self.result_queue = result_queue

        # --- HPO Algorithm params
        self.DUP_CHECK = True
        self._hpo_params = self.hps_info["hpo_params"]
        self._n_params = self._hpo_params["n_params"]
        self._n_steps = self._hpo_params["n_steps"]
        self._eval_key = self._hpo_params["eval_key"]
        self.k = self._hpo_params["k_val"]

        # --- ML Algorithm Params
        self._pbounds = self.hps_info["ml_params"]["pbounds"]

        # --- duplicate params
        self.unique_param_dict = dict()
        self.hash_idx_list = list()
        self.score_list = list()

    ######################################################################
    # HPOptimize
    def optimize(self):
        param_list = list()
        score_list = list()
        # ---- Hyper Parameter Optimize
        for i in range(self._n_steps):
            # --- Generate Candidate parameters
            hyper_param_list = self._generate(param_list, score_list)
            self.LOGGER.info(hyper_param_list)

            # --- Get learning results
            hash_list, score_list = self._learn(i, hyper_param_list)
            self.LOGGER.info("{},{}".format(param_list, score_list))

            # --- Get best parameters
            # global
            self.hash_idx_list += hash_list
            self.score_list += score_list
            param_list, score_list = self._update(self.hash_idx_list, self.score_list)
            self.LOGGER.info("{},{}".format(param_list, score_list))

        # ---- return top-k best parameter and score
        return self._make_best_params(self.hash_idx_list, self.score_list)

    ######################################################################
    # --- abstract function... must implement child class!
    def _check_hpo_params(self):
        raise NotImplementedError

    def _generate(self, param_list, score_list):
        raise NotImplementedError

    # ----- Generate Parameters
    def _generate_param_dict(self, bound_dict):
        param_dict = HPOptimizerUtil.gen_param_dict_to_bound_dict(bound_dict)
        tmp_hash = HPOptimizerUtil.param_dict_to_hash(param_dict)

        if not HPOptimizerUtil.is_duplicate(self.unique_param_dict, tmp_hash):
            self.unique_param_dict[tmp_hash] = {"param_dict": param_dict}
            return param_dict
        else:
            return self._generate_param_dict(bound_dict)

    def _generate_param_dict_list(self, num_params: int) -> List:
        param_dict_list = list()
        for i in range(num_params):
            param_dict = self._generate_param_dict(self._pbounds)
            param_dict_list.append(param_dict)
        return param_dict_list

    def _make_learning_param_dict(self, step, idx, hyper_params):
        temp_dict = dict()
        model_nm = "-".join([self.hps_info["hpo_alg"], self.hps_info["ml_alg"], str(step), str(idx)])
        temp_dict["model_nm"] = model_nm

        return dict(temp_dict, **self.hps_info["ml_params"]["model_param"], **hyper_params)

    def _add_unique_dict(self, param_dict_list) -> None:
        for param_dict in param_dict_list:
            tmp_hash = HPOptimizerUtil.param_dict_to_hash(param_dict)
            self.unique_param_dict[tmp_hash] = {"param_dict": param_dict}

    # ----- Generate Parameters END

    def _learn(self, step, hyper_param_list, dup_exclude=False):
        # ---- get learning results
        hash_list = list()
        score_list = list()

        curr_job = 0
        for idx, hyper_params in enumerate(hyper_param_list):
            temp_hash = HPOptimizerUtil.param_dict_to_hash(hyper_params)

            # duplicated parameters
            if HPOptimizerUtil.is_duplicate(self.unique_param_dict, temp_hash):
                _score = self.unique_param_dict.get(temp_hash).get("score", None)
                if _score is not None and dup_exclude is False:
                    hash_list.append(temp_hash)
                    score_list.append(_score)
                    continue

            # make params
            param_dict = self._make_learning_param_dict(step, idx, hyper_params)
            # send job
            self.job_queue.put([temp_hash, param_dict])
            curr_job += 1

        while curr_job > 0:
            results = self.result_queue.get()

            # ---- hash value
            _temp_hash = results.get("hash_value")
            hash_list.append(_temp_hash)

            # ---- score
            score = float(results["results"][-1][self._eval_key])
            score_list.append(score)

            # ---- store history
            _temp_dict = self.unique_param_dict.get(_temp_hash, None)
            _temp_dict["score"] = score
            _temp_dict["results"] = results

            curr_job -= 1

        return hash_list, score_list

    def _sorted_score(self, score_list):
        # -- minimize (default)
        reverse = False
        # --  maximize
        if self._eval_key == "accuracy":
            reverse = True

        sorted_index_list = sorted(range(len(score_list)), key=score_list.__getitem__, reverse=reverse)
        return sorted_index_list

    def _update(self, hash_list, score_list):
        result_params = list()
        result_score = list()

        n_best = np.clip(len(hash_list), 0, self._n_params)

        sorted_index_list = self._sorted_score(score_list)
        for idx in sorted_index_list[:n_best]:
            param_dict = self.unique_param_dict[hash_list[idx]]["param_dict"]
            result_params.append(param_dict)
            result_score.append(score_list[idx])

        return result_params, result_score

    # --- result
    def _make_best_params(self, hash_list, score_list):
        best_params_list = list()
        sorted_index_list = self._sorted_score(score_list)
        for idx in sorted_index_list[:self.k]:
            param_dict = self.unique_param_dict[hash_list[idx]]
            best_params_list.append(param_dict)
        return best_params_list
