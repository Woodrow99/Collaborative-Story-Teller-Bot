# 2-Layer-LSTM Model Blueprint:
1. LSTM layer with 128 memory cells and activation tanh.
1. LSTM layer with 128 memory cells and activation tanh.
1. Dropout layer with drop out rate 0.2.
1. Dense layer with 128 neurons and activation ReLU.
1. Dropout layer with drop out rate 0.2.
1. Dense layer with neurons equal to the size of unique characters in the corpus text and activation softmax.
