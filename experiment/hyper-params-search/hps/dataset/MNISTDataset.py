# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.
from typing import Tuple, List

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from hps.dataset.DatasetAbstract import DatasetAbstract


# class : MNISTDataset
class MNISTDataset(DatasetAbstract):
    @classmethod
    def load(cls, dim=1, supervised=True) -> Tuple[List, List]:
        dataset, ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=supervised,
            with_info=True,
        )

        ds_train = cls.to_list(dataset[0], dim, supervised)
        ds_test = cls.to_list(dataset[1], dim, supervised)
        return ds_train, ds_test

    @classmethod
    def reshape(cls, dim, data: np.array) -> np.array:
        if dim == 1:
            return cls.reshape_1d(data)
        raise NotImplementedError

    @staticmethod
    def reshape_1d(data: np.array) -> np.array:
        return data.reshape([-1]) / 255

    # @staticmethod
    # def get():
    #     return tfds.load(
    #         'mnist',
    #         split=['train', 'test'],
    #         shuffle_files=True,
    #         as_supervised=True,
    #         with_info=True,
    #     )
    #     # return (ds_train, ds_test), ds_info
    #
    # @staticmethod
    # def normalize_img(image, label):
    #     return tf.cast(image, tf.float32) / 255., label
    #
    # @staticmethod
    # def normalize_img_2d(image, label):
    #     return tf.cast(tf.reshape(image, [-1, 28]), tf.float32) / 255., label
    #
    # @staticmethod
    # def normalize_img_1d(image, label):
    #     return tf.cast(tf.reshape(image, [-1]), tf.float32) / 255., label
    #
    # @classmethod
    # def make_dataset(cls, normalize_fn):
    #     (ds_train, ds_test), ds_info = cls.get()
    #
    #     # train dataset
    #     ds_train = ds_train.map(
    #         normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     ds_train = ds_train.cache()
    #     ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    #     ds_train = ds_train.batch(128)
    #     ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    #
    #     # test dataset
    #     ds_test = ds_test.map(
    #         normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     ds_test = ds_test.batch(128)
    #     ds_test = ds_test.cache()
    #     ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    #
    #     return ds_train, ds_test
    #
    # @classmethod
    # def get_tf_dataset_3d(cls):
    #     return cls.make_dataset(cls.normalize_img)
    #
    # @classmethod
    # def get_tf_dataset_2d(cls):
    #     return cls.make_dataset(cls.normalize_img_2d)
    #
    # @classmethod
    # def get_tf_dataset_1d(cls):
    #     return cls.make_dataset(cls.normalize_img_1d)


if __name__ == '__main__':
    # (train, test), info = MNISTDataset.get()
    #
    # print(tfds.as_numpy(train))
    #
    # print(MNISTDataset.get_tf_dataset_1d())
    # print(MNISTDataset.get_tf_dataset_2d())
    # print(MNISTDataset.get_tf_dataset_3d())
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import datetime
    start = datetime.datetime.now()
    train, test = MNISTDataset().load()
    end = datetime.datetime.now()
    print("time : ", (end - start))
