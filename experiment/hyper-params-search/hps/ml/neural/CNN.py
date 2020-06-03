# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.

import tensorflow as tf

from hps.utils.TensorFlowUtils import TensorFlowUtils
from hps.ml.TensorFlowAbstract import TensorFlowAbstract

# class : CNN
class CNN(TensorFlowAbstract):
    def _check_hyper_params(self, param_dict):
        _param_dict = dict()
        try:
            _param_dict["input_units"] = int(param_dict["input_units"])
            _param_dict["output_units"] = int(param_dict["output_units"])
            _param_dict["hidden_units"] = list(map(int, str(param_dict["hidden_units"]).split(",")))
            _param_dict["initial_weight"] = float(param_dict["initial_weight"])
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            _param_dict["dropout_prob"] = float(param_dict["dropout_prob"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
            _param_dict["filter_sizes"] = list(map(int, str(param_dict["filter_sizes"]).split(",")))
            _param_dict["pool_sizes"] = list(map(int, str(param_dict["pool_sizes"]).split(",")))
            _param_dict["num_filters"] = int(param_dict["num_filters"])
            _param_dict["pooling_fn"] = str(param_dict["pooling_fn"])
            _param_dict["conv_fn"] = str(param_dict["conv_fn"])
        except:
            raise
        return _param_dict

    def build(self):
        ## model
        model_nm = self.param_dict["model_nm"]
        algorithm_type = self.param_dict["algorithm_type"]

        ## DNN Parameter Setting
        input_units = self.param_dict["input_units"]
        output_units = self.param_dict["output_units"]
        hidden_units = self.param_dict["hidden_units"]
        initial_weight = self.param_dict.get("initial_weight", 0.1)
        act_fn = TensorFlowUtils.get_active_fn(self.param_dict.get("act_fn", "relu"))
        dropout_prob = self.param_dict.get("dropout_prob", 0.1)
        optimizer_fn = TensorFlowUtils.get_optimizer_fn(self.param_dict.get("optimizer_fn", "adam"))
        learning_rate = self.param_dict.get("learning_rate", 0.1)
        filter_sizes = self.param_dict["filter_sizes"]
        pool_sizes = self.param_dict["pool_sizes"]
        num_filters = self.param_dict["num_filters"]
        pooling_fn = self.param_dict["pooling_fn"]
        conv_fn = self.param_dict["conv_fn"]

        pooling = TensorFlowUtils.get_pooling_fn(pooling_fn)
        conv = TensorFlowUtils.get_conv_fn(conv_fn)
        conv_stride = None
        pooling_stride = None

        ### keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=(input_units,), name= model_nm + '_X')
        self.model.add(self.inputs)

        if "1D" in conv_fn:
            conv_stride = 1
            pooling_stride = 2
            self.model.add(
                tf.keras.layers.Reshape(
                    (input_units, 1),
                    name="{}_input_reshape".format(model_nm)
                )
            )

        for i, filter_size in enumerate(filter_sizes):
            # Convolution Layer
            conv_cls = conv(
                kernel_size=filter_size,
                filters=num_filters,
                strides=conv_stride,
                padding="SAME",
                activation=act_fn,
                name="{}_conv_{}".format(model_nm, i)
            )
            self.model.add(conv_cls)

            # Pooling Layer
            pooled_cls = pooling(
                pool_size=pool_sizes[i],
                strides=pooling_stride,
                padding='SAME',
                name="{}_pool_{}".format(model_nm, i))
            self.model.add(pooled_cls)

        flatten_cls = tf.keras.layers.Flatten()
        self.model.add(flatten_cls)
        self.model.add(
            tf.keras.layers.Dropout(
                dropout_prob
            )
        )

        units = TensorFlowUtils.get_units(self.model.output_shape[1], hidden_units, output_units)

        # Multi-Layer Perceptron
        TensorFlowUtils.mlp_block(
            self.model, units, act_fn, dropout_prob, initial_weight,
            model_nm+"mlp", algorithm_type
        )
        self.model.summary()
        TensorFlowUtils.compile_model(self.model, algorithm_type, output_units, optimizer_fn, learning_rate)


if __name__ == '__main__':
    parameters = {
        ## model parameters
        "model_nm" : "CNN-test",
        "algorithm_type" : "classifier",
        "job_type" : "learn",
        ## learning parameters
        "global_step" : "10",
        "early_type": "none",
        "min_step": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
        ## algorithm parameters
        "input_units": "784",
        "output_units": "10",
        "hidden_units" : "100, 200, 100",
        "dropout_prob" : "0.1",
        "optimizer_fn" : "Adam",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "act_fn" : "tanh",
        "filter_sizes" : "2,2",
        "pool_sizes" : "2,2",
        "num_filters" : "16",
        "pooling_fn" : "Max1D",
        "conv_fn" : "Conv1D",
    }
    cnn = CNN(parameters)
    cnn.build()

    from hps.dataset.MNISTDataset import MNISTDataset
    ds_learn, ds_test = MNISTDataset.get_tf_dataset_1d()

    cnn.learn(ds_learn)
    print(cnn.predict(ds_test))