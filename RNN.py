import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np

with open('corpus_text/anne-of-green-gables_full.json') as json_file:
    text = json.load(json_file)
    words = []
    for sent in text:
        for word in sent:
            words.append(word)
    setWords = sorted(list (set (words)))
    print(len(setWords))
    print(setWords)

    print('total words:', len(words))
    wordIndices = {}
    indicesWord = {}
    num = 0
    for word in words:
        if word not in wordIndices.keys():
            wordIndices[word] = num
            indicesWord[num] = word
            num =+ 1

sequence_max = 50
sequences = []
nextWord = []
for i in range(0, len(words)-50, 51):  #generates sequence-nextWord pairs
    sequences.append(words[i:i+sequence_max])
    nextWord.append(words[i+sequence_max])

x = np.zeros((len(sequences), sequence_max, len(setWords)),  dtype =np.bool)
y = np.zeros((len(sequences), len(setWords)), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for t, word in enumerate(sequence):
        x[i, t, wordIndices[word]] = 1
    y[i, wordIndices[nextWord[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(sequence_max, len(setWords))))
model.add(Dense(len(setWords), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(x, y, batch_size=128, epochs=2)

model.save('annemodel.h5')


