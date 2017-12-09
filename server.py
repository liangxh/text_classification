# -*- coding: utf-8 -*-
import json
import numpy as np
import os
from flask import Flask

port = 5000


def register_blueprints(app, blueprint_prefix_list):
    for blueprint, prefix in blueprint_prefix_list:
        app.register_blueprint(blueprint)


if __name__ == '__main__':
    app = Flask(__name__)

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        from data_service.view import dataset

        dataset.init_data()

        blueprint_prefix_list = [
            (dataset.blueprint, '')
        ]
        register_blueprints(app, blueprint_prefix_list)

    app.run(debug=True, port=port)
