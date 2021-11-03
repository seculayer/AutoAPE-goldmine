import tensorflow as tf
import numpy as np
from hps.common.Common import Common

class PredictionResults(object):
    def __init__(self, prediction:list, label):
        self._TP = float()
        self._TN = float()
        self._FP = float()
        self._FN = float()
        self.LOGGER = Common.LOGGER.getLogger()

        self._sorted_score_list = list()
        self._prediction = prediction
        self._label = label

        self.confusion_matrix()

    def average_bestn(self, accuracy_list)->dict: # sorted_score_list, n = 1, 5, total
        total_sum = 0
        tmp_sum = 0 
        accuracy_list.sort()

        for i in range(5):
            tmp_sum += accuracy_list[i]

        for _, v in enumerate(accuracy_list):
            total_sum += v

        best = accuracy_list[0]
        best5 = tmp_sum/5
        best_total = total_sum/len(accuracy_list)
        result_dict = {
            "best" : best,
            "best5" : best5,
            "best_total" : best_total
        }
        return result_dict


    def n_params_accuracy(self, threshold=0.9)->int:
        # TODO : calculate the n_params to reach the threshold
        for i, v in enumerate(self._sorted_score_list):
            if v >= threshold:
                return i
        return -1

    ####################################Confusion Matrix###########################

    def confusion_matrix(self):
        labels = np.array(self._label)#.flatten()
        cvt_labels = np.zeros((len(labels), (np.max(labels)+1))).tolist()

        for k, v in enumerate(labels):
            cvt_labels[k][v]=1

        prediction = np.argmax(self._prediction, axis=1)
        cvt_prediction = np.zeros((len(prediction),(np.max(labels)+1))).tolist()
        for k, v in enumerate(prediction):
            cvt_prediction[k][v]=1

        np_labels = np.array(cvt_labels)
        np_prediction = np.array(cvt_prediction)

        self._TP  = tf.math.count_nonzero(np_labels*np_prediction)
        self._TN  = tf.math.count_nonzero((np_prediction-1)*(np_labels-1))
        self._FP  = tf.math.count_nonzero(np_prediction*(np_labels-1))
        self._FN  = tf.math.count_nonzero((np_prediction-1)*np_labels)
        self.LOGGER.info("TP : {}, TN : {}, FP : {}, FN : {}".format(self._TP, self._TN, self._FP, self._FN))

    def accuracy(self)->float:
        accuracy = (self._TP+self._TN)/(self._TP+self._TN+self._FP+self._FN)
        self.LOGGER.info("accuracy : {}".format(accuracy))
        return accuracy

    def f1_score(self)->float:
        score = (2*self._TP)/(2*self._TP+self._FP+self._FN)
        self.LOGGER.info("f1 score : {}".format(score))
        return score

    def precision(self)->float:
        precision = self._TP/(self._TP+self._FP)
        self.LOGGER.info("precision : {}".format(precision))
        return precision

    def recall(self)->float:
        recall = self._TP/(self._TP+self._FN)
        self.LOGGER.info("recall : {}".format(recall))
        return recall