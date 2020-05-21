# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

# class : class_name
class TensorFlowAbstract(object):
    def __init__(self, param_dict):
        pass

    def _check_parameter(self, param_dict):
        raise NotImplementedError

    def _build(self):
        raise NotImplementedError