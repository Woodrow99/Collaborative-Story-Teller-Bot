import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
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
for i in range(0, len(words)-sequence_max, sequence_max + 1):  #generates sequence-nextWord pairs
    sequence = words[i:i+sequence_max]
    nextWord = words[i+sequence_max]
    dataX.append([wordIndices[word] for word in sequence])
    dataY.append(wordIndices[nextWord])


X = np.reshape(dataX, (len(dataX), sequence_max, 1))
X = X / float(len(setWords))

y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(X, y, batch_size=128, epochs=50)

model.save('annemodel.h5')

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([indicesWord[value] for value in pattern]), "\"")
for i in range(1000):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(setWords))
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = indicesWord[index]
	seq_in = [indicesWord[value] for value in pattern]
	sys.stdout.write(result + ' ')
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print( "\nDone.")











