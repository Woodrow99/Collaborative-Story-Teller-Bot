import json

with open('anne-of-green-gables_full.json') as json_file:
    text = json.load(json_file)
    words = []
    for sent in text:
        for word in sent:
            words.append(word)
    sorted(words)
    print('total words:', len(words))
    wordIndices = {}
    indicesWord = {}
    num = 0
    for word in words:
        if word not in wordIndices.keys():
            wordIndices[word] = num
            indicesWord[word] = num
            num =+ 1

sequence_max = 50
sequences = []
nextWord = []
for i in range(0, len(words)-50, 51):  #generates sequence-nextWord pairs
    sequences.append(words[i:i+sequence_max])
    nextWord.append(words[i+sequence_max])

