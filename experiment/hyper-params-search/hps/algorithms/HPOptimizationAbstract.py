
import numpy as np
import json

from hps.common.Constants import Constants
from hps.common.Common import Common
from hps.ml.ProcessManager import ProcessManager

# class : HPOptimizationAbstract
class HPOptimizationAbstract(object):
    def __init__(self, hps_info, **kwargs):
        ### eyeCloudAI framework objects
        self.LOGGER = Common.LOGGER.getLogger()

        ### basic variables
        self.hps_info = hps_info
        self.dataset_nm = self.hps_info["dataset"]

        ### HPO Algorithm params
        self.DUP_CHECK = True
        self._hpo_params = self.hps_info["hpo_params"]
        self._n_params = self._hpo_params["n_params"]
        self._n_steps = self._hpo_params["n_steps"]
        self._eval_key = self._hpo_params["eval_key"]
        self.k = self._hpo_params["k_val"]

        ### ML Algorithm Params
        self._pbounds = self.hps_info["ml_params"]["pbounds"]

        ### duplicate params
        self.unique_param_dict = dict()
        self.hash_idx_list = list()
        self.score_list = list()

        print(self.hps_info)
    ##################
    # HPOptimize
    def optimize(self):
        param_list = list()
        score_list = list()
        ## Hyper Parameter Optimize
        for i in range(self._n_steps):
            ### Generate Candidate parameters
            hyper_param_list = self._generate(param_list, score_list)
            self.LOGGER.info(hyper_param_list)

            ### Get learning results
            hash_list, score_list = self._learn(i, hyper_param_list)
            self.LOGGER.info("{},{}".format(param_list, score_list))

            ### get best parameters
            # global
            self.hash_idx_list += hash_list
            self.score_list += score_list
            param_list, score_list = self._update(self.hash_idx_list, self.score_list)
            self.LOGGER.info("{},{}".format(param_list, score_list))

        ## return top-k best parameter and score
        return self._make_best_params(self.hash_idx_list, self.score_list)

    ###############
    ### abstract function... must implement child class!
    def _check_hpo_params(self):
        raise NotImplementedError

    def _generate(self, param_list, score_list):
        raise NotImplementedError

    ###### Generate Parameters
    @staticmethod
    def _generate_int_list_params(bound_data):
        tmp_list = list()
        for i in range(np.random.randint(1,bound_data[0])):
            tmp_list.append(int(np.random.randint(1,bound_data[1])))
        return ",".join(map(str, tmp_list))

    @staticmethod
    def _generate_single_params(bound_data):
        if type(bound_data[1]) == int :
            return int(np.random.randint(bound_data[0], bound_data[1]))
        elif type(bound_data[1]) == float :
            return np.random.uniform(bound_data[0], bound_data[1])
        else :
            return str(np.random.choice(bound_data))

    def _generate_param(self, key):
        if key == 'hidden_units' or key == "filter_sizes" or key == "pool_sizes":
            # TODO : first value of hidden, filter, pool must be above 1
            return self._generate_int_list_params(self._pbounds[key])
        else :
            return self._generate_single_params(self._pbounds[key])

    def _generate_param_dict(self, dup_check=True):
        param_dict = dict()
        for _ , key in enumerate(self._pbounds):
             param_dict[key] = self._generate_param(key)

        if not self._check_duplicated_param_dict(self.unique_param_dict, param_dict) and dup_check:
            return param_dict
        else :
            return self._generate_param_dict()

    def _generate_param_dict_list(self, num_params):
        param_dict_list = list()
        for _ in range(num_params):
            param_dict = self._generate_param_dict(dup_check=self.DUP_CHECK)
            param_dict_list.append(param_dict)
        return param_dict_list
    ###### Generate Parameters END

    ### DUPLICATE
    @staticmethod
    def _param_dict_to_hash(param_dict):
        result_str = ""
        for k, v in sorted(dict(param_dict).items()):
            result_str +="{}{}".format(k, v)
        return str(hash(result_str))

    def _check_duplicated_param_dict(self, unique_dict, param_dict):
        tmp_hash = self._param_dict_to_hash(param_dict)
        if unique_dict.get(tmp_hash, None) is None:
            unique_dict[tmp_hash] = {"param_dict" : param_dict}
            return False
        else:
            return True

    def _remove_duplicate_params(self, param_dict_list):
        new_params = list()
        unique_dict = dict()
        for param_dict in param_dict_list:
            if self._check_duplicated_param_dict(unique_dict, param_dict):
                new_params.append(param_dict)
        return new_params

    ###
    def _make_learning_param_dict(self, step, idx, hyper_params):
        temp_dict = dict()
        model_nm = "-".join([self.hps_info["hpo_alg"], self.hps_info["ml_alg"], str(step), str(idx)])
        temp_dict["model_nm"] = model_nm

        return dict(temp_dict, **self.hps_info["ml_params"]["model_param"], **hyper_params)

    def _learn(self, step, hyper_param_list, dup_exclude=False):
        ## get learning results
        hash_list = list()
        score_list = list()

        proc_manager = ProcessManager(self.dataset_nm)

        for idx, hyper_params in enumerate(hyper_param_list):
            _temp_hash = self._param_dict_to_hash(hyper_params)

            # duplicated parameters
            if self._check_duplicated_param_dict(self.unique_param_dict, hyper_params):
                _score = self.unique_param_dict.get(_temp_hash).get("score", None)
                if  _score is not None and dup_exclude == False:
                    hash_list.append(_temp_hash)
                    score_list.append(_score)
                    continue

            # make params
            param_dict = self._make_learning_param_dict(step, idx, hyper_params)
            proc_manager.append(_temp_hash, self.hps_info.get("ml_alg"), param_dict)

        proc_manager.start()
        proc_manager.join()

        for results in proc_manager.get_results():
            ## hash value
            _temp_hash = results.get("hash_value")
            hash_list.append(_temp_hash)

            ## score
            score = float(results["results"][-1][self._eval_key])
            score_list.append(score)

            ### store history
            _temp_dict = self.unique_param_dict[_temp_hash]
            _temp_dict["score"] = score
            _temp_dict["results"] = results

        return hash_list, score_list

    def _sorted_score(self, score_list):
        ## minimize (default)
        reverse = False
        ## maximize
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

    #### result
    def _make_best_params(self, hash_list, score_list):
        best_params_list = list()
        sorted_index_list = self._sorted_score(score_list)
        for idx in sorted_index_list[:self.k]:
            param_dict = self.unique_param_dict[hash_list[idx]]
            best_params_list.append(param_dict)
        return best_params_list
