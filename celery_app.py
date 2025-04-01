<<<<<<< HEAD
# celery_app.py
from celery import Celery
from config import Config

celery = Celery(__name__, broker=Config.CELERY['broker_url'], backend=Config.CELERY['result_backend'])
=======
# celery_app.py
from celery import Celery
from config import Config

celery = Celery(__name__, broker=Config.CELERY['broker_url'], backend=Config.CELERY['result_backend'])
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
celery.conf.update(Config.CELERY)