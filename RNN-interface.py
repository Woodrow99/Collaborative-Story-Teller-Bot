from pickle import load
from keras.models import load_model
from numpy import array
from keras.utils.np_utils import to_categorical
import os

def main():
    og_dir = os.getcwd()
    os.chdir(og_dir + "/corpus")
    batchSet = os.listdir()
    os.chdir(og_dir)
    num = 1
    print("Select a batch for the RNN to train on by number:")
    for fileName in batchSet:
        print(str(num) + ": " + fileName)
        num += 1
    userInput = 0
    while not (0 < userInput < len(batchSet)):
        userInput = int(input("Selection: "))
    os.chdir(og_dir)
    RunBatch("corpus/" + batchSet[userInput - 1], load_model("model.h5"), load(open('tokenizer.pkl', 'rb')))

def LoadDoc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def RunBatch(fileName, myModel, tokenizer):
    text = LoadDoc(fileName)
    lines = text.split('\n')
    vocabSize = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = array(sequences)
    X = sequences[:, :-1]
    y = sequences[:,-1]
    y = to_categorical(y, num_classes=vocabSize)
    print("Running batch for", fileName)
    myModel.fit(X, y, batch_size=128, epochs=75, shuffle=True)
    myModel.save('model.h5')

main()