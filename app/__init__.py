<<<<<<< HEAD
# app/__init__.py
from flask import Flask
from flask_restful import Api
from config import Config
from .api.chat import ChatAPI
from flask_cors import CORS
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    CORS(app)

    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = False #Keep

    api = Api(app)
    api.add_resource(ChatAPI, '/api/chat')

=======
# app/__init__.py
from flask import Flask
from flask_restful import Api
from config import Config
from .api.chat import ChatAPI
from flask_cors import CORS
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    CORS(app)

    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = False #Keep

    api = Api(app)
    api.add_resource(ChatAPI, '/api/chat')

>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
    return app