# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hprs.core.algorithms.hpo.GeneticAlgorithm import GeneticAlgorithm
# class : HPOptimizerFactory
class HPOptimizerFactory(object):
    @staticmethod
    def create(hpo_dict):
        hpo_alg = hpo_dict["hpo_alg"]
        if hpo_alg == "GA":
            # TODO : Check init & remove get_ga_params
            ga = GeneticAlgorithm(hprs_info=hpo_dict)
            return ga
        else:
            return GeneticAlgorithm()
