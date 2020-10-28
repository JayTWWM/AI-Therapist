from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from wtforms import StringField,PasswordField,SubmitField,BooleanField,TextAreaField
from wtforms.validators import DataRequired,Length,Email,EqualTo,email_validator,ValidationError
from emotions.models import User
from flask_login import current_user

class RegistrationForm(FlaskForm):
    username = StringField('Username',validators=[DataRequired(),Length(min=2,max=20)])
    email = StringField('Email',validators=[DataRequired(),Email(message='Enter valid email',granular_message=True)])
    password = PasswordField('Password',validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                    validators=[DataRequired(),EqualTo('password',message='Passwords dont match')])
    medical_history = StringField('Medical History',validators=[DataRequired()])
    mood_history = StringField('Mood History',validators=[DataRequired()])
    submit = SubmitField('Sign Up')

    def validate_username(self,username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists! Try another username')

    def validate_email(self,email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already exists! Try another email')

class LoginForm(FlaskForm):
    #username = StringField('Username',validators=[DataRequired(),Length(min=2,max=20)])
    email = StringField('Email',validators=[DataRequired(),Email()])
    password = PasswordField('Password',validators=[DataRequired()])
    #confirm_password = PasswordField('Confirm Password',
    #                               validators=[DataRequired(),EqualTo('password',message='Passwords dont match')])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class SubmitText(FlaskForm):
    text = StringField('Text',validators=[DataRequired(),Length(min=2)])
    submit_text = SubmitField('Submit')

    