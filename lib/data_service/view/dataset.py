# -*- coding: utf-8 -*-
import json
import numpy as np
from flask import Blueprint

blueprint = Blueprint('dataset', __name__)
prefix = '/api/dataset'
_json_data = {}


def load_imdb():
    imdb = np.load('/home/anonymous/.keras/datasets/imdb.npz')

    def _merge_to_pair_list(x, y):
        n = len(x)
        return [(x[i], y[i]) for i in range(n)]

    def _preprocess(x, y):
        return json.dumps(_merge_to_pair_list(x, y))

    return {
            'train': _preprocess(imdb['x_train'], imdb['y_train']),
            'test': _preprocess(imdb['x_test'], imdb['y_test'])
        }


def init_data():
    global _json_data
    _json_data['imdb'] = load_imdb()


@blueprint.route('/<dataset_name>/<subset>', methods=['GET'])
def get_dataset(dataset_name, subset):
    global _json_data
    return _json_data[dataset_name][subset]
