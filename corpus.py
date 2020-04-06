import os
import json
import nltk
import re
from nltk.corpus import PlaintextCorpusReader
from nltk import PorterStemmer
from nltk.corpus import stopwords 


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
blanks = ['']

def create_corpus():
    og_dir = os.getcwd()
    os.chdir(og_dir + "/raw")
    wordLists = PlaintextCorpusReader(os.getcwd(), '.*')
    textList = wordLists.fileids()

    normSentsList = []

    for i in textList:
        
        opened = open(i,'r')
        readable = opened.read()
        string_list = readable.split()

        for word in string_list:
            word = word.lower()
            word = re.sub('\W+','', word)
            normSentsList.append(word)

    start = 0
    end = 50
    sequenceList = []
    while end <= len(normSentsList):
        sequenceList.append(normSentsList[start:end])
        start += 1 
        end += 1
    fle = open("training_sequences", "w+")
    fle. write(json.dumps(sequenceList))
            
create_corpus()