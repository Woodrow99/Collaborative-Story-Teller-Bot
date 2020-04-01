import json

from keras_preprocessing.text import Tokenizer
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers import LSTM
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.utils import np_utils
import sys

with open('corpus_text/anne-of-green-gables_full.json') as json_file:
    text = json.load(json_file)
    words = []
    for sent in text:
        for word in sent:
            words.append(word)
    setWords = sorted(list(set(words)))
    print('total words:', len(words))
    wordIndices = {}
    indicesWord = {}
    num = 0
    for word in setWords:
        wordIndices[word] = num
        indicesWord[num] = word
        num += 1

sequence_max = 50
dataX = []
dataY = []
for i in range(0, len(words) - sequence_max, sequence_max + 1):  # generates sequence-nextWord pairs
    sequence = words[i:i + sequence_max]
    nextWord = words[i + sequence_max]
    dataX.append([wordIndices[word] for word in sequence])
    dataY.append(wordIndices[nextWord])

X = np.array(dataX)
y = np.array(dataY)
y = np_utils.to_categorical(dataY, num_classes=len(setWords))
print(X)
print(y)


def CreateModel():
    model = Sequential()
    model.add(Embedding(len(setWords), 50))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=100)
    model.save('annemodel.h5')


def GenerateText(model, pattern):
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ' '.join([indicesWord[value] for value in pattern]), "\"")
    for i in range(1000):
        x = np.array(pattern)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        print(index)
        result = indicesWord[index]
        seq_in = [indicesWord[value] for value in pattern]
        sys.stdout.write(result + ' ')
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")


# CreateModel()
GenerateText(load_model("annemodel.h5"), dataX[np.random.randint(0, len(dataX) - 1)])
