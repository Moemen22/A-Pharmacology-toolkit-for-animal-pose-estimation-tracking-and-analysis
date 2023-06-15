import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers import SimpleRNN



sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

df = pd.read_csv("C:\\Users\\Moeme\\PycharmProjects\\TestLSTM\\merged_file2.csv")
#df = df.dropna(how='any')

df['Label'].value_counts().plot(kind='bar', title='Training examples by activity type');

df.head()

N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['X'].values[i: i + N_TIME_STEPS]
    ys = df['Y'].values[i: i + N_TIME_STEPS]
    zs = df['Z'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['Label'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)

"""### np.array(segments).shape"""

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

reshaped_segments.shape

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

model = keras.Sequential()
model.add(keras.Input(shape=(N_TIME_STEPS,N_FEATURES)))
model.add(Dropout(0.5))
model.add(SimpleRNN(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit and evaluate a model
def evaluate_modela(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 400, 428
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(SimpleRNN(100, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(N_TIME_STEPS,N_FEATURES)))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = list()
for r in range(10):
    score = evaluate_modela(X_train,y_train,X_test, y_test)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))