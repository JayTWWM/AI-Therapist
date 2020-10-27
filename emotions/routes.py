import secrets,os,sys,librosa,glob,flask,ktrain
import tensorflow as tf
from playsound import playsound
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# CUDA_VISIBLE_DEVICES=""

import pandas as pd
import tkinter
from tkinter import *
from flask import render_template,url_for,flash,redirect,request,abort
from emotions import app,db,bcrypt,login_manager
from emotions.forms import RegistrationForm,LoginForm,SubmitText
from emotions.models import User
from flask_login import login_user,current_user,logout_user,login_required
from keras.models import load_model,model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 as cv
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.compat.v1.Session(config=config)

mp = {'happy':'https://manybooks.net/categories','sad':'https://www.youtube.com/watch?v=F9wbogYwTVM'}

face_classifier = cv.CascadeClassifier('D:\Machine Learning\EmotionRecognizer\haarcascade_frontalface_default.xml')
model12 = load_model('D:\Machine Learning\EmotionRecognizer\Emotion_little_vgg.h5')

model1 = ktrain.load_predictor('D:\Machine Learning\EmotionRecognizer\emotions\mood_text')
classes12 = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# json_file = open('D:\Machine Learning\EmotionRecognizer\emotions\model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
loaded_model = load_model("D:\Machine Learning\EmotionRecognizer\emotions\saved_models\Emotion_Voice_Detection_Model.h5")
# print("Loaded model from disk")

# lister = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
c1 = ['neutral','calm','happy','surprised']


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
model = load_model('D:\Machine Learning\EmotionRecognizer\emotions\chatbot_model.h5')
import json
import random
intents = json.loads(open('D:\Machine Learning\EmotionRecognizer\emotions\intents.json').read())
words = pickle.load(open('D:\Machine Learning\EmotionRecognizer\emotions\words.pkl','rb'))
classes = pickle.load(open('D:\Machine Learning\EmotionRecognizer\emotions\classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


@app.route('/')
@app.route('/register',methods=["POST","GET"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        
        user = User(username=form.username.data,email=form.email.data,password=hashed_password,
        medical_history=form.medical_history.data,mood_history=form.mood_history.data)
        db.session.add(user)
        db.session.commit()
        flash(f'Your account has been created!','success')
        return redirect(url_for('login'))
    return render_template('register.html',title='Register',form=form)

@app.route('/login',methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is not None and bcrypt.check_password_hash(user.password,form.password.data):
            login_user(user,remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home')) 
        else:
            flash(f'Invalid login credentials','danger')
            #return redirect(url_for('login'))
    return render_template('login.html',title='Login',form=form)
    
def suggest_remedy(mood):
    if mood in ('Happy','Neutral','Surprise'):
        return redirect_to_remedy(mood)
    else:
        return redirect_to_remedy(mood)

def redirect_to_remedy(mood):
    return render_template(mp.get(mood,'https://www.youtube.com/watch?v=F9wbogYwTVM'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html',title='About Page')

@app.route('/predict')
def predict_mood():
    final_label = None
    cap = cv.VideoCapture(0)
    got = False
    while True:
        ret,frame = cap.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        
        for x,y,w,h in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)
            
            if(np.sum([roi_gray])!=0):
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                
                preds = model12.predict(roi)[0]
                label = classes12[preds.argmax()]
                label_position = (x,y)
                final_label = label
                # got = True
                # break
                cv.putText(frame,label,label_position,cv.FONT_HERSHEY_COMPLEX,2,(0,255,0))
            else:
                cv.putText(frame,'No Face Found',(20,60),cv.FONT_HERSHEY_COMPLEX,2,(0,0,255))
        # if got:
        #     break
        cv.imshow('Emotion Detector',frame)
        if cv.waitKey(1) == 13:
            break
    cap.release()
    cv.destroyAllWindows()
    # print("Done")
    print(final_label)
    if final_label in ('Happy','Neutral','Surprise'):
        return render_template("happy.html")
    else:
        return render_template("sad.html")

@app.route("/audio", methods=['POST', 'GET'])
def audio():
    fs = 44100
    seconds = 2.5
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write('D:\Machine Learning\EmotionRecognizer\emotions\output10.wav', fs, myrecording)
    X, sample_rate = librosa.load('D:\Machine Learning\EmotionRecognizer\emotions\output10.wav')
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    livedf2 = mfccs
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame()
    x = np.expand_dims(livedf2, axis=2)
    x = np.expand_dims(x, axis=0)
    predictions = loaded_model.predict_classes(x)
    mood = convert_class_to_emotion(predictions)
    print(mood)
    if mood in c1:
        return render_template("happy.html")
    else:
        return render_template("sad.html")
    

@app.route('/logout',methods=["GET","POST"])
def logout():
    logout_user()
    return redirect(url_for('signup'))

@app.route('/mood_text',methods=['GET','POST'])
def mood_text():
    form=SubmitText()
    if form.validate_on_submit():
        text = form.text.data
        prediction=model1.predict(text)
        print(prediction)
        if prediction in ('joy','love','surprise'):
            return render_template("happy.html")
        else:
            return render_template("sad.html")

    return render_template('text_page.html',title='Submit_Text',form=form)

def record_mood(mood):
    if form.validate_on_submit():
        db.session.add(mood)
        db.session.commit()
        flash(f'Your mood has been recorded!','success')
        return redirect(url_for('login'))
    return render_template('register.html',title='Register',form=form)

@app.route('/chatbot')
def talk_to_chatbot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot_response(userText))

def convert_class_to_emotion(pred):        
    label_conversion = {'0': 'neutral',
                        '1': 'calm',
                        '2': 'happy',
                        '3': 'sad',
                        '4': 'angry',
                        '5': 'fearful',
                        '6': 'disgust',
                        '7': 'surprised'}

    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label