# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.utils.keras.callback.LearnResultCallback import LearnResultCallback
from hps.utils.keras.callback.EarlyStopCallback import EarlyStopCallback


# class : TensorFlowAbstract
class TensorFlowAbstract(object):
    def __init__(self, param_dict):
        self.param_dict = self._check_parameter(param_dict)
        self.model = None
        self.inputs = None
        self.stopped_epoch = None

    def _check_parameter(self, param_dict):
        return dict(
            self._check_model_params(param_dict),
            **self._check_learning_params(param_dict),
            **self._check_hyper_params(param_dict)
        )

    @staticmethod
    def _check_model_params(param_dict):
        _param_dict = dict()
        try:
            ### learning parameters
            _param_dict["model_nm"] = str(param_dict["model_nm"])
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            # _param_dict["job_type"] = str(param_dict["job_type"])
        except Exception as e:
            raise

        return _param_dict

    @staticmethod
    def _check_learning_params(param_dict):
        _param_dict = dict()
        try:
            _param_dict["global_step"] = int(param_dict["global_step"])
            if _param_dict["global_step"] < 1:
                _param_dict["global_step"] = 1
            _param_dict["early_type"] = param_dict["early_type"]
            if _param_dict["early_type"] != "none":
                _param_dict["min_step"] = int(param_dict["min_step"])
                _param_dict["early_key"] = param_dict["early_key"]
                _param_dict["early_value"] = float(param_dict["early_value"])
        except Exception as e:
            raise

        return _param_dict

    def _check_hyper_params(self, param_dict):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def learn(self, dataset, verbose=0):
        ## callbacks
        result_callback = LearnResultCallback(global_sn=self.param_dict.get("global_sn", "0"))
        early_stop_callback = EarlyStopCallback(self.param_dict)

        self.model.fit(
            dataset, verbose=verbose, epochs=self.param_dict.get("global_step", 1),
            callbacks=[result_callback, early_stop_callback],
        )

        self.stopped_epoch = early_stop_callback.get_stopped_epoch()
        return result_callback.get_result()

    def predict(self, dataset):
        return self.model.predict(dataset)
