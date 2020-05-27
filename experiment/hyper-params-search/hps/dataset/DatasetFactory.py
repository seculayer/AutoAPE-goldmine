# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.dataset.MNISTDataset import MNISTDataset

# class : DatasetFactory
class DatasetFactory(object):
    @staticmethod
    def create(data_nm, dim=1):
        data_nm = data_nm.lower()
        if data_nm == "mnist":
            if dim == 1:
                return MNISTDataset.get_tf_dataset_1d()

if __name__ == '__main__':
    name = "MNIST"
    ds_train, ds_test = DatasetFactory.create(name)
    print(ds_train, ds_test)