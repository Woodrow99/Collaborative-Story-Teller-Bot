import json
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    with open(filename) as json_file:
        text = json.load(json_file)
        lines = []
        for seq in text:
            line = ''
            for word in seq:
                line += word + ' '
            lines.append(line)
    return lines


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


# load cleaned text sequences
in_filename = 'corpus_text/anne-of-green-gables_full.json'
lines = load_doc(in_filename)
seq_length = len(lines[0]) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# select a seed text
seed = lines[randint(0, len(lines))]
print(seed + '\n')

# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed, 50)
print(generated)