# -*- coding: utf-8 -*-
"""Emotion_text.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14ln4Q1ihhIXyKmeCptSUnJJG0xYffFuh
"""

import os
os.chdir("drive/My Drive/SAD")
!unzip 605165_1085454_bundle_archive.zip

import pandas as pd
df_test = pd.read_csv("/content/drive/My Drive/SAD/test.txt",delimiter=';', header=None, names=['sentence','label'])
df_train= pd.read_csv("/content/drive/My Drive/SAD/train.txt",delimiter=';', header=None, names=['sentence','label'])
df_val=pd.read_csv("/content/drive/My Drive/SAD/val.txt",delimiter=';', header=None, names=['sentence','label'])

df.head()

len(df)

df_test['label'].unique()

df_final = pd.concat([df_test,df_train,df_train])

len(df_final)

df_train_final=df_final[0:17000]

df_train_final.shape

df_test_final=df_final[17000:34000]

df_test_final.shape

df_test_final.head()

df_train_final.head()

!pip install ktrain

import tensorflow as tf
import ktrain
from ktrain import text

import numpy as np

x_train =np.asarray(df_train_final['sentence'])
y_train = np.asarray(df_train_final['label'])
x_test = np.asarray(df_test_final['sentence'])
y_test = np.asarray(df_test_final['label'])

trn, val, preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                          x_test=x_test, y_test=y_test,
                                          class_names=df_train_final['label'].unique(),
                                          preprocess_mode='distilbert',
                                          maxlen=350)

text.print_text_classifiers()

model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)

learner.fit_onecycle(3e-5, 6)

p = ktrain.get_predictor(model, preproc)

p.predict('Since a politician never believes what he says, he is surprised when others believe him')

p.save('/content/drive/My Drive/bert')

predictor_load = ktrain.load_predictor('/content/drive/My Drive/SAD/bert')

predictor_load.predict('im feeling quite sad and sorry for myself but ill snap out of it soon')

predictor_load.predict('i am feeling very blessed today that they share such a close bond')

predictor_load.predict('i am feeling very blessed today that they share such a close bond')
