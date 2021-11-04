# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.common.Constants import Constants
from hps.algorithms.ga.GeneticAlgorithm import GeneticAlgorithm


# class : HPOptimizerFactory
class HPOptimizerFactory(object):
    @staticmethod
    def create(hpo_dict, job_queue, result_queue):
        hpo_alg = hpo_dict["hpo_alg"]
        return {
            Constants.HPO_GA: GeneticAlgorithm(hps_info=hpo_dict,
                                               job_queue=job_queue,
                                               result_queue=result_queue)
        }.get(hpo_alg, None)
