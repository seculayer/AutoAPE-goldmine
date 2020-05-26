# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.

import tensorflow as tf

from hps.utils.TensorFlowUtils import TensorFlowUtils
from hps.ml.TensorFlowAbstract import TensorFlowAbstract

# class : DNN
class DNN(TensorFlowAbstract):
    def _check_hyper_params(self, param_dict):
        _param_dict = dict()
        try:
            _param_dict["input_units"] = int(param_dict["input_units"])
            _param_dict["output_units"] = int(param_dict["output_units"])
            _param_dict["hidden_units"] = list(map(int, str(param_dict["hidden_units"]).split(",")))
            _param_dict["initial_weight"] = float(param_dict["initial_weight"])
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            _param_dict["dropout_prob"] = float(param_dict["dropout_prob"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
        except:
            raise
        return _param_dict

    def build(self):
        ## model
        model_nm = self.param_dict["model_nm"]
        algorithm_type = self.param_dict["algorithm_type"]

        ## DNN Parameter Setting
        input_units = self.param_dict["input_units"]
        output_units = self.param_dict["output_units"]
        hidden_units = self.param_dict["hidden_units"]
        initial_weight = self.param_dict.get("initial_weight", 0.1)
        act_fn = TensorFlowUtils.get_active_fn(self.param_dict.get("act_fn", "relu"))
        dropout_prob = self.param_dict.get("dropout_prob", 0.1)
        optimizer_fn = TensorFlowUtils.get_optimizer_fn(self.param_dict.get("optimizer_fn", "adam"))
        learning_rate = self.param_dict.get("learning_rate", 0.1)
        units = TensorFlowUtils.get_units(input_units, hidden_units, output_units)

        ### keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=(units[0],), name= model_nm + '_X')

        # Multi-Layer Perceptron
        TensorFlowUtils.mlp_block(
            self.model, units, act_fn, dropout_prob, initial_weight,
            model_nm+"mlp", algorithm_type
        )
        TensorFlowUtils.compile_model(self.model, algorithm_type, output_units, optimizer_fn, learning_rate)


if __name__ == '__main__':
    parameters = {
        ## model parameters
        "model_nm" : "DNN-test",
        "alg_sn" : "0",
        "global_sn" : "0",
        "algorithm_type" : "classifier",
        "job_type" : "learn",
        ## learning parameters
        "global_step" : "100",
        "early_type": "none",
        "min_step": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
        ## algorithm parameters
        "input_units": "2",
        "output_units": "2",
        "hidden_units" : "5,4,3",
        "dropout_prob" : "0.5",
        "optimizer_fn" : "Adam",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "act_fn" : "ReLU",
    }
    dnn = DNN(parameters)
    dnn.build()

    import numpy as np
    x = np.array([[-1., -1.], [-2., -1.], [1., 1.], [2., 1.]])
    y = np.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    dnn.learn(x, y)