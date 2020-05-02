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
    generated_text = ''
    sequence = seed_text
    generated_text += sequence

    for i in range(gen_length):
        x_seed = np.zeros((1, len(sequence), len(chars)))
        for t, char in enumerate(sequence):
            x_seed[0, t, char_to_indice[char]] = 1.

        lst_of_probs = model.predict(x_seed, verbose=0)[0]
        pred_index = temp_func(lst_of_probs, temp)
        pred_char = indice_to_char[pred_index]

        sequence = sequence[:] + pred_char
        generated_text += pred_char

    return generated_text

def temp_func(lst_of_probs, temperature=1.0):
    """
    :param preds: A numpy array containing the probablities of all characters in a given sequence
    :param temperature: an int that the resulting numpy array is divided by. Default is 1.0
    :return: an integer that represents the most likely character divided by a temperature
    """
    np.seterr(divide='ignore')
    float_probs = np.asarray(lst_of_probs).astype('float64')
    log_probs = np.log(float_probs) / temperature
    exp_probs = np.exp(log_probs)
    predictions = exp_probs / np.sum(exp_probs)
    solution_lst = np.random.multinomial(1, predictions, 1)
    return np.argmax(solution_lst)

#main()
