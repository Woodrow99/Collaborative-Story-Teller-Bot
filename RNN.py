from pickle import dump

from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.optimizers import Adam
from numpy import array
from keras.utils.np_utils import to_categorical
import os
from keras_preprocessing.text import Tokenizer
from nltk.corpus import PlaintextCorpusReader


def LoadDoc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

og_dir = os.getcwd()
os.chdir(og_dir + "/corpus")
vocabText = LoadDoc("vocab.txt")
vocabList = vocabText.split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocabList)
seqLength = 50
vocabSize = len(tokenizer.word_index) + 1
os.chdir(og_dir)

def CreateModel():
    model = Sequential()
    model.add(Embedding(vocabSize, seqLength))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(vocabSize, activation='softmax'))
    print(model.summary())
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.save("model.h5")
CreateModel()
dump(tokenizer, open('tokenizer.pkl', 'wb'))


