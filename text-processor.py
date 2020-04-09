import os
import string
from random import shuffle

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r' ,encoding= "utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load document
og_dir = os.getcwd()
os.chdir(og_dir + "/raw")
batchTotal = 25
batches = []
allSequences = []
for i in range(0, batchTotal+1):
	batches.append(list())
currentBatch  = 0
setTokens = []
for fileName in os.listdir():
	doc = load_doc(fileName)
	tokens = clean_doc(doc)
	for token in tokens:
		if token not in setTokens:
			setTokens.append(token)
	# organize into sequences of tokens
	length = 50 + 1
	for i in range(length, len(tokens)):
		if currentBatch == batchTotal:
			currentBatch = 0
		else:
			currentBatch += 1
		# select sequence of tokens
		seq = tokens[i-length:i]
		# convert into a line
		line = ' '.join(seq)
		# store
		batches[currentBatch].append(line)
		allSequences.append(line)

# save sequences to file
os.chdir(og_dir + '/corpus')
for i in range(0, batchTotal + 1):
	out_filename = 'batch' + str(i) + '.txt'
	save_doc(batches[i], out_filename)
	save_doc(allSequences, 'complete_corpus.txt')
save_doc(setTokens, 'vocab.txt')

