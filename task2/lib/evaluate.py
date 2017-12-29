# -*- coding: utf-8 -*-
import numpy as np


def f1(precision, recall):
    return (2.0 * precision * recall) / (precision + recall)


def score(labels_gold, labels_predict):
    truth_dict = dict()
    output_dict_correct = dict()
    output_dict_attempted = dict()

    for label_gold, label_predict in zip(labels_gold, labels_predict):
        if label_gold not in truth_dict:
            truth_dict[label_gold] = 1
        else:
            truth_dict[label_gold] += 1

        if label_predict == label_gold:
            if label_predict not in output_dict_correct:
                output_dict_correct[label_predict] = 1
            else:
                output_dict_correct[label_predict] += 1

        if label_predict not in output_dict_attempted:
            output_dict_attempted[label_predict] = 1
        else:
            output_dict_attempted[label_predict] += 1

    n_label = len(truth_dict)
    attempted_total = 0
    correct_total = 0
    gold_occurrences_total = 0
    f1_total = 0

    for label in truth_dict:
        gold_occurrences = truth_dict[label]
        if label in output_dict_attempted:
            attempted = output_dict_attempted[label]
        else:
            attempted = 0

        if label in output_dict_correct:
            correct = output_dict_correct[label]
        else:
            correct = 0
        if attempted != 0:
            precision = (correct * 1.0) / attempted
            recall = (correct * 1.0) / gold_occurrences
            if precision != 0.0 or recall != 0.0:
                f1_total += f1(precision, recall)
        attempted_total += attempted
        correct_total += correct
        gold_occurrences_total += gold_occurrences

    macro_f1 = f1_total / (n_label * 1.0)
    precision_total_micro = (correct_total * 1.0) / attempted_total
    recall_total_micro = (correct_total * 1.0) / gold_occurrences_total

    if precision_total_micro != 0.0 or recall_total_micro != 0.0:
        micro_f1 = f1(precision_total_micro, recall_total_micro)
    else:
        micro_f1 = 0.0

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'precision': precision_total_micro,
        'recall': recall_total_micro
    }
