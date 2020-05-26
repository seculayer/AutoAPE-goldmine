
import numpy as np
import json

from hps.common.Constants import Constants
from hps.common.Common import Common


class HPOptimizationAbstract(object):
    def __init__(self, hprs_info, **kwargs):
        ### eyeCloudAI framework objects
        self.LOGGER = Common.LOGGER.getLogger()

        ### basic variables
        self.hprs_info = hprs_info

        ### HPO Algorithm params
        self._hpo_params = self.hprs_info["hpo_params"]
        self._n_params = self._hpo_params["n_params"]
        self._n_steps = self._hpo_params["n_steps"]
        self._eval_key = self._hpo_params["eval_key"]
        self.k = self._hpo_params["k_val"]

        ### ML Algorithm Parmas
        self._pbounds = self.hprs_info["ml_params"]["pbounds"]

        ### duplicate params
        self.unique_param_dict = dict()
        self.hash_idx_list = list()
        self.score_list = list()

    ##################
    # HPOptimize
    def optimize(self):
        hprs_id = self.hprs_info["hprs_id"]
        ml_alg = self.hprs_info["ml_alg"]
        param_list = list()
        score_list = list()
        ## Hyper Parameter Optimize
        for i in range(self._n_steps):
            ### Generate Candidate parameters
            hyper_param_list = self._generate(param_list, score_list)
            self.LOGGER.info(hyper_param_list)
            ### Get learning results
            hash_list, score_list = self._learn(hprs_id, ml_alg, i, hyper_param_list)
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

    def _generate_param_dict(self):
        param_dict = dict()
        for _ , key in enumerate(self._pbounds):
             param_dict[key] = self._generate_param(key)

        if not self._check_duplicated_param_dict(self.unique_param_dict, param_dict):
            return param_dict
        else :
            return self._generate_param_dict()

    def _generate_param_dict_list(self, num_params):
        param_dict_list = list()
        for _ in range(num_params):
            param_dict = self._generate_param_dict()
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

    ### LEARNING
    @staticmethod
    def _insert_db_to_candidate_param(hprs_id, hprs_hist_no, ml_alg, hyper_params):
        insert_data = {
            "hprs_id" : hprs_id,
            "hprs_hist_no" : hprs_hist_no,
            "ml_alg": ml_alg,
            "hyper_params" : json.dumps(hyper_params)
        }
        HPOptimizeDAO().insert_hpo_hist(insert_data)

    @staticmethod
    def _make_candidate_param_monitor_thread(hprs_id, hprs_hist_no):
        _th = HPOWorkerMonitor(data={"hprs_id" : hprs_id, "hprs_hist_no" : hprs_hist_no})
        _th.start()
        return _th

    def _learn(self, hprs_id, ml_alg, step, hyper_param_list):
        monitor_list = list()
        for idx, hyper_params in enumerate(hyper_param_list):
            ## MAKE hprs_hist_no
            hprs_hist_no = str(hprs_id) + str(step) + str(idx)
            ## INSERT DB
            self._insert_db_to_candidate_param(hprs_id, hprs_hist_no, ml_alg, hyper_params)
            ## CREATE DB MONITORING THREAD
            monitor_list.append(self._make_candidate_param_monitor_thread(hprs_id, hprs_hist_no))

        ## get learning results
        hash_list = list()
        score_list = list()
        for monitor in monitor_list:
            monitor.join()

            param_dict = monitor.get_param_dict()
            results = monitor.get_results()
            if results is not None and param_dict is not None:
                ## param_dict
                _temp_hash = self._param_dict_to_hash(param_dict)
                hash_list.append(_temp_hash)

                ## score
                alg_sn = 0
                score = float(results[alg_sn][-1][self._eval_key])
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
