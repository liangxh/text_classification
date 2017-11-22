# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from flask import Flask

port = 5000


def register_blueprints(app, blueprint_prefix_list):
    for blueprint, prefix in blueprint_prefix_list:
        app.register_blueprint(blueprint, url_prefix=prefix)


if __name__ == '__main__':
    app = Flask(__name__)

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        from lib.data_service.view import dataset
        dataset.init_data()

        blueprint_prefix_list = [
            (dataset.blueprint, dataset.prefix)
        ]
        register_blueprints(app, blueprint_prefix_list)

    app.run(debug=True, port=port)
