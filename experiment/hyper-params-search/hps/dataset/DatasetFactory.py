# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.dataset.MNISTDataset import MNISTDataset
from hps.dataset.Cifar10Dataset import Cifar10Dataset


# class : DatasetFactory
class DatasetFactory(object):
    @staticmethod
    def load(data_nm, dim=1):
        data_nm = data_nm.lower()
        if data_nm == "mnist":
            try:
                return MNISTDataset.load(dim, True)
            except Exception as e:
                print("dimension is max 3")
                return None, None

        if data_nm == "cifar10":
            if dim == 1:
                return Cifar10Dataset.get_tf_dataset_1d()

    @staticmethod
    def get(data_nm):
        data_nm = data_nm.lower()


if __name__ == '__main__':
    name = "MNIST"
    ds_train, ds_test = DatasetFactory.create(name, 4)
    print(ds_train, ds_test)
