# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

# --- keras neural network
from hps.ml.TensorFlowAbstract import TensorFlowAbstract
from hps.ml.neural.DNN import DNN
from hps.ml.neural.CNN import CNN
from hps.ml.neural.RNN import RNN


# class : MLAlgorithmFactory
class MLAlgorithmFactory(object):
    @staticmethod
    def create(algorithm_name, param_dict) -> TensorFlowAbstract:
        if algorithm_name == "DNN":
            return DNN(param_dict)
        elif algorithm_name == "CNN":
            return CNN(param_dict)
        elif algorithm_name == "RNN":
            return RNN(param_dict)

        raise NotImplementedError
