<<<<<<< HEAD
# app/models/brand.py
from app import db

class Brand(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    text = db.Column(db.Text, nullable=False)
    __table_args__ = (
        db.UniqueConstraint('name', 'text', name='unique_brand_text'),
=======
# app/models/brand.py
from app import db

class Brand(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    text = db.Column(db.Text, nullable=False)
    __table_args__ = (
        db.UniqueConstraint('name', 'text', name='unique_brand_text'),
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
    )