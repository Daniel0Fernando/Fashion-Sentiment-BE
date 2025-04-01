<<<<<<< HEAD
# app/services/celery_tasks.py
from celery_app import celery
# REMOVE: from app.services.scraper import scrape_multiple_brands # No longer needed

@celery.task(bind=True)
def train_model_async(self):
    """Background task to train the model."""
    from app.services.trainer import train_model

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Training started'})
        result = train_model()
        self.update_state(state='SUCCESS', meta={'status': result})
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'status': str(e)})
=======
# app/services/celery_tasks.py
from celery_app import celery
# REMOVE: from app.services.scraper import scrape_multiple_brands # No longer needed

@celery.task(bind=True)
def train_model_async(self):
    """Background task to train the model."""
    from app.services.trainer import train_model

    try:
        self.update_state(state='PROGRESS', meta={'status': 'Training started'})
        result = train_model()
        self.update_state(state='SUCCESS', meta={'status': result})
        return result
    except Exception as e:
        self.update_state(state='FAILURE', meta={'status': str(e)})
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
        raise