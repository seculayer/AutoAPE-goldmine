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
        #cls.LOGGER.info("label_count : {}".format(label.value_counts()))
        balanced_data, balanced_label = cls.balancing_data(data, label)
        normalized_data = cls.normalize(balanced_data)
        cls.LOGGER.info("nan_count : {}".format(np.isnan(normalized_data).sum()))

        train, test,train_label, test_label = cls.split_data(normalized_data, balanced_label)
        ds_train, ds_test, label_train, label_test = cls.make_dataset(train,test, train_label, test_label)
        cls.LOGGER.info("spec of train:{}, test:{}, train_label:{}, test_label:{}".format(np.shape(ds_train),np.shape(ds_test),np.shape(label_train),np.shape(label_test)))
        return ds_train, label_train, ds_test, label_test

    @classmethod
    def load_data(cls)-> (pd.DataFrame, pd.DataFrame):
        filenames = [i for i in glob.glob(join("/../../../../cicids", "*.pcap_ISCX.csv"))]
        combined_csv = pd.concat([pd.read_csv(f,dtype=object,skipinitialspace=True) for f in filenames],sort=False,)
        data = combined_csv.rename(columns=lambda x: x.strip()) #strip() : delete \n & space
        #data.replace([np.inf, -np.inf], 0)

        cls.LOGGER.info("{}".format(data['Label'].value_counts()))
        #if data["Label"] == 1. :
        #    data["Label"] = 1
        #else :
        #    data["Label"] = 0

        label = data["Label"]
        #cls.LOGGER.info("label count: {}".format(label.value_counts()))
        # encoded_label = cls.encode_label(label.values)
        encoded_label, num = cls.label_encoder(data)
        data.replace('Infinity',0.0, inplace=True)
        data = data.astype(float).apply(pd.to_numeric)
        #encoded_label = cls.ont_hot_encoder(encoded_label, num)
        data.drop("Label", inplace=True, axis=1)

        #data = data.astype(float).apply(pd.to_numeric)
        data.fillna(0, inplace=True)
        cls.LOGGER.info("data  : {}".format(data))
        encoded_label = encoded_label.astype("float")
        #cls.LOGGER.info("label : {}".format(encoded_label))
        # data["Label"]=encoded_label

        #cls.LOGGER.info("label : {}".format(encoded_label.unique()))
        data = data.values

        #encoded_label = cls.ont_hot_encoder(encoded_label, num)
        return data, encoded_label

    @classmethod
    def make_dataset(cls, train:np.ndarray, test:np.ndarray, train_label:np.ndarray, test_label:np.ndarray)->(list, list):
        cls.LOGGER.info("train :{}".format(train.shape))
        cls.LOGGER.info("train_label :{}".format(train_label.shape))

        ds_train_list = train.tolist()
        ds_test_list = test.tolist()
        label_train_list = train_label.tolist()
        label_test_list = test_label.tolist()

        return ds_train_list, ds_test_list, label_train_list, label_test_list

    @classmethod
    def split_data(cls, dataset:np.ndarray, label:np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        data_size = int(len(label))
        train_size = int(0.65 * data_size)
        split_data= np.split(dataset,[train_size, data_size])
        split_label= np.split(label, [train_size, data_size])
        return split_data[0], split_data[1], split_label[0], split_label[1]

    @classmethod
    def encode_label(cls, y_str):
        labels_d = cls.make_lut(np.unique(y_str))
        y = [labels_d[y_str_i] for y_str_i  in y_str]
        return np.array(y)


    @classmethod
    def label_encoder(cls, data):
        #cls.LOGGER.info("target length : {}".format(data["Label"].value_counts()))
        target = data["Label"]
        num_uni = np.unique(target)#, axis = 0)
        num = num_uni.shape[0]
        for idx in range(len(num_uni)):
            target[target == num_uni[idx]] = idx
        #encoding = np.eye(num)[target]
        #return encoding
        return target, num

    @classmethod
    def ont_hot_encoder(cls, label_target, num):
        label_target = np.array(label_target).astype(int)
        encoding = np.eye(num)[label_target]
        return encoding


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
        data = tf.keras.utils.normalize(data)
        #data = data.astype(np.float32)
        #mask = (data==-1)
        #data[mask]=0
        #mean_i = np.mean(data)#,axis=0)
        #min_i = np.min(data)#,axis=0)
        #max_i = np.max(data)#,axis=0)
        #eps = 1e-10
        #data = (data-min_i)/(max_i-min_i+eps)
        #data[mask] = 0
        return data

    @classmethod
    def balancing_data(cls, data, label, seed=2):
        np.random.seed(seed)

        unique, counts = np.unique(label, return_counts=True)
        mean_samples_per_class = int(round(np.mean(counts)))
        cls.LOGGER.info("Unique : {}, Counts : {}, mean : {}".format(unique, counts, mean_samples_per_class))
        new_data = np.empty((0,78), dtype=float) # np.empty((0, 78), dtype = float)
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

    #print("train = {}".format(train.shape))
    #print("test = {}".format(test.shape))
    #print("train_label = {}".format(train_label.shape))
    #print("test_label = {}".format(test_label.shape))

