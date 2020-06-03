# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.

import tensorflow as tf

from hps.utils.TensorFlowUtils import TensorFlowUtils
from hps.ml.TensorFlowAbstract import TensorFlowAbstract

# class : RNN
class RNN(TensorFlowAbstract):
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
            _param_dict["cell_units"] = int(param_dict["cell_units"])
            _param_dict["rnn_cell"] = str(param_dict["rnn_cell"])
            _param_dict["seq_length"] = int(param_dict["seq_length"])
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
        cell_units = self.param_dict["cell_units"]
        rnn_cell = self.param_dict["rnn_cell"]
        seq_length = self.param_dict["seq_length"]

        cell = TensorFlowUtils.get_rnn_cell(rnn_cell)

        ### keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=(seq_length * input_units, ), name= model_nm + '_X')
        self.model.add(self.inputs)

        self.model.add(
            tf.keras.layers.Reshape(
                (seq_length, input_units),
                name="{}_input_reshape".format(model_nm)
            )
        )

        self.model.add(
            cell(
                units=cell_units,
                activation=act_fn,
                dropout=dropout_prob,
                name="{}_cell".format(model_nm),
            )
        )

        units = TensorFlowUtils.get_units(cell_units, hidden_units, output_units)

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
        "model_nm" : "RNN-test",
        "algorithm_type" : "classifier",
        "job_type" : "learn",
        ## learning parameters
        "global_step" : "10",
        "early_type": "none",
        "min_step": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
        ## algorithm parameters
        "input_units": "28",
        "output_units": "10",
        "hidden_units" : "100, 200, 100",
        "dropout_prob" : "0.1",
        "optimizer_fn" : "Adam",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "act_fn" : "tanh",
        "cell_units" : "4",
        "rnn_cell" : "LSTM",
        "seq_length" : "28"
    }
    rnn = RNN(parameters)
    rnn.build()

    from hps.dataset.MNISTDataset import MNISTDataset
    ds_learn, ds_test = MNISTDataset.get_tf_dataset_1d()

    rnn.learn(ds_learn)
    print(rnn.predict(ds_test))