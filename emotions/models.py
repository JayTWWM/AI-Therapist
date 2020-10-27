from datetime import datetime
from emotions import db,login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model,UserMixin):
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.String(20),nullable=False,unique=True)
    email = db.Column(db.String(120),nullable=False,unique=True)
    #image_file = db.Column(db.String(20),nullable=False,default='default.jpg')
    password = db.Column(db.String(60),nullable=False)
    #posts = db.relationship('Post',backref='author',lazy=True)
    medical_history = db.Column(db.String(200))
    mood_history = db.Column(db.String(200))
    
    def __repr__(self):
        return f"User('{self.username}','{self.email}')"

class Prediction:
    def __init__(self,user_input_type,user_input,user):
        self.user_input_type = user_input_type
        self.user_input = user_input
        self.user = user

class Remedy:
    def __init__(self,remedy_type,remedy_priority,remedy_link):
        self.remedy_type = remedy_type
        self.remedy_priority = remedy_priority
        self.remedy_link = remedy_link

