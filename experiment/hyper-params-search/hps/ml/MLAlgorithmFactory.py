# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

### keras neural network
from hps.ml.neural.DNN import DNN

# class : MLAlgorithmFactory
class MLAlgorithmFactory(object):
    @staticmethod
    def create(algorithm_name, param_dict):
        if algorithm_name == "DNN":
            return DNN(param_dict)

        raise NotImplementedError
