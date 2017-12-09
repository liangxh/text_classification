# -*- coding: utf-8 -*-
import json
import os
import requests
import server
from data_service.view import dataset

url_prefix = 'http://127.0.0.1:{}'.format(server.port)


def get_dataset(name, subset):
    url = os.path.join(url_prefix + dataset.blueprint.url_prefix, '{}/{}'.format(name, subset))
    res = requests.get(url)
    return json.loads(res.text.encode('utf8'))


def _test():
    get_dataset('imdb', 'train')


if __name__ == '__main__':
    _test()
