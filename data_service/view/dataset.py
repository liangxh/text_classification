# -*- coding: utf-8 -*-
import os
import json
import cPickle
from flask import Blueprint

blueprint = Blueprint('dataset', __name__, url_prefix='/api/dataset')
_json_data = {}


def _load_imdb():
    filename = os.path.join(os.environ['HOME'], '.keras/datasets/imdb_full.pkl')
    train_x_y, test_x_y = cPickle.load(open(filename, 'r'))

    return {
        'train': json.dumps(zip(*train_x_y)),
        'test': json.dumps(zip(*test_x_y))
    }


def init_data():
    global _json_data
    _json_data['imdb'] = _load_imdb()


@blueprint.route('/<dataset_name>/<subset>', methods=['GET'])
def get_dataset(dataset_name, subset):
    global _json_data
    return _json_data[dataset_name][subset]
