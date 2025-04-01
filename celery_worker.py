# celery_worker.py
from celery_app import celery
from app.services.celery_tasks import train_model_async