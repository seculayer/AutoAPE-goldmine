import numpy as np
import pandas as pd
from os.path import join
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from hps.common.Common import Common
from hps.dataset.DatasetAbstract import DatasetAbstract

class CICIDSDataset(DatasetAbstract):
    LOGGER = Common.LOGGER.getLogger()

    @classmethod
    def get_dataset(cls):
        data, label = cls.load_data()
        balanced_data, balanced_label = cls.balancing_data(data, label)
        normalized_data = cls.normalize(data)
        train, test, train_label, test_label = cls.split_data(normalized_data, balanced_label)
        cls.LOGGER.info("spec of train:{}, test:{}, train_label:{}, test_label:{}".format(np.shape(train),np.shape(test),np.shape(train_label),np.shape(test_label)))
        ds_train, ds_test, label_train, label_test = cls.make_dataset(train,test, train_label, test_label)
        return ds_train, ds_test, label_train, label_test

    @classmethod
    def reshape(cls, dim:int, data: np.array):
        return data

    @classmethod
    def load_data(cls)-> (pd.DataFrame, pd.DataFrame):
        filenames = [i for i in glob.glob(join("/../../../../dltest", "*.pcap_ISCX.csv"))]
        combined_csv = pd.concat([pd.read_csv(f,dtype=object,skipinitialspace=True) for f in filenames],sort=False,)
        data = combined_csv.rename(columns=lambda x: x.strip()) #strip() : delete \n & space

        label = data["Label"]
        data.drop("Label", inplace=True, axis=1)
        encoded_label = cls.encode_label(label.values)

        data.dropna(axis=0)
        data = data.astype(float).apply(pd.to_numeric)
        data = data.values
        return data, encoded_label

    @classmethod
    def make_dataset(cls, train:np.ndarray, test:np.ndarray, train_label:np.ndarray, test_label:np.ndarray)->(list, list):
        ds_train = tf.data.Dataset.from_tensor_slices(train)
        ds_test = tf.data.Dataset.from_tensor_slices(test)
        label_train = tf.data.Dataset.from_tensor_slices(train_label)
        label_test = tf.data.Dataset.from_tensor_slices(test_label)

        ds_train = ds_train.shuffle(200, seed=2).batch(50, drop_remainder=True)
        ds_test = ds_test.shuffle(200, seed=2).batch(50, drop_remainder=True)
        label_train = label_train.shuffle(200, seed=2).batch(50, drop_remainder=True)
        label_test = label_test.shuffle(200, seed=2).batch(50, drop_remainder=True)
        cls.LOGGER.info("before to list : {}".format(ds_train))


        # ds_train_list = cls.to_list(dataset=ds_train, dim=1, supervised=False)
        # ds_test_list = cls.to_list(dataset=ds_test, dim=1, supervised=False)
        # label_train_list = cls.to_list(dataset=label_train, dim=1, supervised=False)
        # label_test_list = cls.to_list(dataset=label_test, dim=1, supervised=False)
        return ds_train_list, ds_test_list, label_train_list, label_test_list

    @classmethod
    def split_data(cls, dataset:np.ndarray, label:np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        data_size = int(len(label)/100)
        train_size = int(0.65 * data_size)
        split_data= np.split(dataset,[train_size, data_size])
        split_label= np.split(label, [train_size, data_size])
        return split_data[0], split_data[1], split_label[0], split_label[1]

    @classmethod
    def encode_label(cls, y_str):
        labels_d = cls.make_lut(np.unique(y_str))
        y = [labels_d[y_str] for y_str  in y_str]

        return np.array(y)

    @classmethod
    def make_lut(cls, unique):
        #make dictionary
        category = dict()
        counter=0
        for char in unique:
            category[char] = counter
            counter+=1
        return category

    @classmethod
    def normalize(cls, data):
        data = data.astype(np.float32)

        mask = (data==-1)
        data[mask]=0
        mean_i = np.mean(data,axis=0)
        min_i = np.min(data,axis=0)
        max_i = np.max(data,axis=0)

        data = (data-min_i)/(max_i-min_i)

        data[mask] = 0
        return data

    @classmethod
    def balancing_data(cls, data, label, seed=2):
        np.random.seed(seed)

        unique, counts = np.unique(label, return_counts=True)
        mean_samples_per_class = int(round(np.mean(counts)))
        cls.LOGGER.info("Unique : {}, Counts : {}, mean : {}".format(unique, counts, mean_samples_per_class))
        new_data = np.empty((0,78), dtype=float)
        new_label = np.empty(0, dtype=int)

        for _,c in enumerate(unique):
            temp_data = data[label==c]
            indices = np.random.randint(0, len(temp_data), size=mean_samples_per_class)
            new_data = np.concatenate((new_data, temp_data[indices]), axis=0)
            temp_label = np.ones(mean_samples_per_class, dtype=int)*c
            new_label = np.concatenate((new_label, temp_label), axis=0)

        cls.LOGGER.info("spec of balanced data:{}, label : {}".format(np.shape(new_data), np.shape(new_label)))

        return new_data,new_label

if __name__ == '__main__' :

    train, test, train_label, test_label= CICIDSDataset.get_dataset()

    print("train = {}".format(train))
    print("test = {}".format(test))
    print("train_label = {}".format(train_label))
    print("test_label = {}".format(test_label))
    print("{}".format(test_label))

