"""
File Name: RNN.py

Author: Isaac Zeimetz

Description: This program is a recurrent neural network with 4 layers:
two LSTM of size 128 and two dense layers, one of size 128, and the other
being the size of the total unique characters in the corpus text.
This file primarily builds the recurrent neural network alongside all of
the extra files it needs to generate text.

note: requires tensorflow, keras, pickle, and matplotlib.pyplot to run
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from helper_functions import load_doc, preprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

#global variables
OG_DIR = os.getcwd()

#functions
def main():
    seq_len = 50
    text = build_text()
    chars, char_to_indice, indice_to_char = dictionary_builder(text)
    sequences, next_chars =sequence_maker(seq_len, 3, text)
    x, y = to_numpy_array(seq_len, sequences, next_chars, chars, char_to_indice)
    model, history1 = build_model(len(chars), x, y)
    plot_data(history1)
    model_name = str(input("Please give the model a name: "))
    save_model(char_to_indice, indice_to_char, model, model_name, chars)

def build_model(chars_len, x, y):
    """
    :param chars_len: the length of the unique characters in the corpus text
    :param x: a numpy array of sequences size 49. Used for training
    :param y: a numpy array of single characters. Used for validation of training
    :return model: an object that represents the fully trained model
    :return history: a dictionary containing data of the results of the RNN's training
    """
    #builds the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, chars_len), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dropout(.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(chars_len, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #trains the model
    history = model.fit(x, y, batch_size=128, epochs=50)
    return model, history

def build_text():
    """
    :return text: the entirety of the corpus text which is encoded as string
    """
    text = ''
    os.chdir(OG_DIR + "/raw")
    for file_name in os.listdir():
        raw_file = load_doc(file_name)
        raw_file = preprocess(raw_file)
        text += raw_file
    os.chdir(OG_DIR)
    return text

def dictionary_builder(text):
    """
    :param text: text is a string, in this case the corpus text
    :return chars: a list of all unique characters that appear in the corpus text
    :return char_to_indice: a dictionary that maps a character to a specific indidce
    :return indice_to_char: a dictionary that maps an indice to a specific character
    """
    chars = sorted(list(set(text)))
    char_to_indice = {}
    indice_to_char = {}
    # places unique characters into dictionaries where they are represented by numbers
    for i in range(0, len(chars)):
        char_to_indice[chars[i]] = i
        indice_to_char[i] = chars[i]
    return chars, char_to_indice, indice_to_char

def plot_data(history):
    """
    :param history: a dictionary containing data of the results of the RNN's training
    :return: nothing
    """
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def save_model(char_to_indice, indice_to_char, model, model_name, chars):
    """
    :param char_to_indice: a dictionary that maps a character to a specific indidce
    :param indice_to_char: a dictionary that maps an indice to a specific character
    :param model: an object that represents the fully trained model
    :param chars: a list of all unique characters that appear in the corpus text
    :return: nothing
    """
    os.makedirs(OG_DIR + "/models/" + model_name)
    os.chdir(OG_DIR + "/models/" + model_name)

    indice_dict = open("indice_to_char.pkl", "wb")
    pickle.dump(indice_to_char, indice_dict)
    indice_dict.close()

    char_dict = open("char_to_indice.pkl", "wb")
    pickle.dump(char_to_indice, char_dict)
    char_dict.close()

    char_file = open("chars.txt", "w", encoding="utf-8")
    for char in chars[1:]:
        char_file.write(char + "\n")
    char_file.close()

    model.save(model_name + ".h5")

    os.chdir(OG_DIR)

def sequence_maker(seq_len, step_size, text):
    """
    :param seq_len: an integer that determines the size of each sequence
    :param step_size: an integer that determines how much to step from each sequence
    :param text: text is a string, in this case the corpus text
    :return sequences: a list of size 49 lists that will be used for training data
    :return next_chars: a list of single characters used for validation data
    """
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_len, step_size):
        sequences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])
    return sequences, next_chars

def to_numpy_array(seq_len, sequences, next_chars, chars, char_to_indice):
    """
    :param seq_len: an integer that determines the size of each sequence
    :param sequences: a list of size 49 lists that will be used for training data
    :param next_chars: a list of single characters used for validation data
    :param chars: a list of all unique characters that appear in the corpus text
    :param char_to_indice: a dictionary that maps a character to a specific indidce
    :return data_x: a numpy array of sequences size 49. Used for training
    :return data_y: a numpy array of single characters. Used for validation of training
    """
    data_x = np.zeros((len(sequences), seq_len, len(chars)), dtype=np.bool)
    data_y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            data_x[i, t, char_to_indice[char]] = 1
        data_y[i, char_to_indice[next_chars[i]]] = 1
    return data_x, data_y

main()