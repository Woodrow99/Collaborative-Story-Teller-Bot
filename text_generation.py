import json
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from random import randint
import os
import sys

def main():
    in_filename = "raw/the-yellow-wallpaper.txt"
    text = load_doc(in_filename)
    seq_len = 50

    og_dir = os.getcwd()
    os.chdir(og_dir + "/new_models")
    chars_str = load_doc("chars.txt")
    chars = chars_str.splitlines()
    chars.insert(0, "\n")
    #loads the dictionaries in
    char_to_indice = load(open("char_to_indice.pkl", "rb"))
    indice_to_char = load(open("indice_to_char.pkl", "rb"))


    # select a seed text
    start_index = randint(0, len(text) - seq_len)
    seed = text[start_index: start_index + seq_len]
    # loads each model
    for fileName in os.listdir():
        if fileName[-3:] == ".h5":
            model = load_model(fileName)
            print("---------------------------------------------------------------------------------------------------")
            print(fileName)
            print(seed + '\n')
            generate_seq_test(model, char_to_indice, indice_to_char, len(seed), seed, chars)
            print("\n")



# load doc into memory
def load_doc(filename):
    file = open(filename, 'r', encoding="utf-8")
    text = file.read()
    file.close()
    return text


# generate a sequence from a language model
def generate_seq_test(model, char_to_indice, indice_to_char, seq_len, seed_text, chars):
    for diversity in [0.2, 0.5, 0.75, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = seed_text
        generated += sentence

        for i in range(1000):
            x_pred = np.zeros((1, seq_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indice[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indice_to_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def generate_seq(model, char_to_indice, indice_to_char, seq_len, seed_text, chars, length, temp):
    generated = ''
    sentence = seed_text
    generated += sentence

    for i in range(length):
        x_pred = np.zeros((1, seq_len, len(chars)))
        for t, char in enumerate(sentence):
            if t in chars:
                x_pred[0, t, char_to_indice[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temp)
        next_char = indice_to_char[next_index]

        sentence = sentence[1:] + next_char
        generated += next_char

    return generated

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#main()