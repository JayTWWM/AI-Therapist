
import keras
import librosa
import numpy as np
import pandas as pd
from config import EXAMPLES_PATH
from config import MODEL_DIR_PATH


class LivePredictions:

    def __init__(self, file):
        self.file = file
        self.path = MODEL_DIR_PATH + 'Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        X, sample_rate = librosa.load(self.file)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        livedf2 = mfccs
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame()
        # data, sampling_rate = librosa.load(self.file)
        # mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13), axis=0)
        x = np.expand_dims(livedf2, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print( "Prediction is", " ", self.convert_class_to_emotion(predictions))

    @staticmethod
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


if __name__ == '__main__':
    live_prediction = LivePredictions(file=EXAMPLES_PATH + 'output10.wav')
    live_prediction.loaded_model.summary()
    live_prediction.make_predictions()
    live_prediction = LivePredictions(file=EXAMPLES_PATH + '10-16-07-29-82-30-63.wav')
    live_prediction.make_predictions()
