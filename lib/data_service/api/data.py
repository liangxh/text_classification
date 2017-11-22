# -*- coding: utf-8 -*-
import os
import json
import requests
import server
from lib.data_service.view import dataset


url_prefix = 'http://127.0.0.1:{}'.format(server.port)


def get_dataset(name, subset):
    url = os.path.join(url_prefix + dataset.prefix, '{}/{}'.format(name, subset))
    res = requests.get(url)
    return json.loads(res.text.encode('utf8'))
