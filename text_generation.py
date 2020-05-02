import numpy as np

"""
File Name: helper_functions

Author: Isaac Zeimetz

Description: This program defines two functions that help generate text sequences.

note: requires tensorflow, keras, and pickle to run
"""

def generate_seq(model, char_to_indice, indice_to_char, seed_text, chars, gen_length, temp):
    """

    :param model: an object that represents the RNN
    :param char_to_indice: a dictionary with character to number relationship
    :param indice_to_char: a dictionary with number to character relationship
    :param seq_len: The length of a sequence
    :param seed_text: The initial text for the text generation
    :param chars: the list of unique characters in the corpus text
    :param length: an int of characters the generator should produce
    :param temp: an int that the resulting numpy array is divided by
    :return generated:
    """
    generated = ''
    sentence = seed_text
    generated += sentence

    for i in range(gen_length):
        x_pred = np.zeros((1, len(sentence), len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_indice[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temp)
        next_char = indice_to_char[next_index]

        sentence = sentence[:] + next_char
        generated += next_char

    return generated

def sample(preds, temperature=1.0):
    """
    :param preds: A numpy array containing the probablities of all characters in a given sequence
    :param temperature: an int that the resulting numpy array is divided by. Default is 1.0
    :return: an integer that represents the most likely character divided by a temperature
    """
    # helper function to sample an index from a probability array
    np.seterr(divide='ignore')
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#main()
