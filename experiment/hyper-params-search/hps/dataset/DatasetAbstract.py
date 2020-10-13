# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.
from typing import Tuple

import numpy as np
import tensorflow_datasets as tfds


# class : DatasetAbstract
class DatasetAbstract(object):
    @staticmethod
    def load(cls, dim=1, supervised=True):
        raise NotImplementedError

    @staticmethod
    def reshape(dim, data: np.array):
        raise NotImplementedError

    @classmethod
    def to_list(cls, dataset, dim=1, supervised=True):
        features = list()
        labels = list()
        for d in tfds.as_numpy(dataset):
            features.append(cls.reshape(dim, d[0]).tolist())
            if supervised:
                labels.append(d[1])

        if supervised:
            return features, labels
        else:
            return features
