import os
import re
from pickle import load
from keras.engine.saving import load_model

"""
File Name: helper_functions

Author: Isaac Zeimetz

Description: This program has a variety of helper functions necessary for the NLP software to run.

note: requires tensorflow, keras, and pickle to run
"""


def load_doc(filename):
    """
    :param filename: takes in a file name utf-8 format
    :return text: a text file with certain characters replaced with white space.
    """
    file = open(filename, 'r', encoding="utf-8")
    text = file.read()
    file.close()
    return text

def load_game(model_name):
    """
    :param model_name: the name of a file located in the models directory.
    :return model: an object which represents the RNN
    :return char_to_indice: a dictionary with character to number relationship
    :return indice_to_char: a dictionary with number to character relationship
    :return chars: a list of all unique characters in a corpus text
    """
    og_dir = os.getcwd()
    try:
        os.chdir(og_dir + "/models/" + model_name)
    except FileNotFoundError:
        print("Oops! That file does not exist in the models directory.")
        raise
    model = load_model(model_name + ".h5")
    char_to_indice = load(open("char_to_indice.pkl", "rb"))
    indice_to_char = load(open("indice_to_char.pkl", "rb"))
    chars_str = load_doc("chars.txt")
    chars = chars_str.splitlines()
    chars.insert(0, "\n")
    return model, char_to_indice, indice_to_char, chars

def preprocess(text):
    base_text = text.replace('”', '"')
    base_text = base_text.replace('“', '"')
    # sorts out everything that isn't a character, number, or punctuation.
    base_text = re.sub('[^a-zA-Z0-9 \n\.\?\!\-\,\(\)\;\:\'\"]', ' ', base_text)
    return base_text