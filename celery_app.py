# celery_app.py
from celery import Celery
from config import Config

celery = Celery(__name__, broker=Config.CELERY['broker_url'], backend=Config.CELERY['result_backend'])
celery.conf.update(Config.CELERY)