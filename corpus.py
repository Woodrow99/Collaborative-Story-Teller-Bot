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

    for i in textList:
        
        normSentsList = []
        SentList = []
        sentsList = wordLists.sents(i)
        for j in range(0, len(sentsList)):
            temp = sentsList[j]
            SentList.append(temp)
            temp = [word.lower() for word in temp]
            temp = [re.sub('\W+','', word) for word in temp]
            #temp = [word for word in temp if not word in stop_words]
            temp = [word for word in temp if not word in blanks]
            temp = [ps.stem(word) for word in temp]
            normSentsList.append(temp)
        os.chdir(og_dir)
        os.chdir(og_dir + "/corpus_text")
        fle = open(i[:-4] + "_reduce.json", "w+") 
        fle.write(json.dumps(normSentsList))
        fle = open(i[:-4] + "_full.json", "w+") 
        fle.write(json.dumps(SentList))
        os.chdir(og_dir)
        os.chdir(og_dir + "/raw")

    
"""
og_dir = os.getcwd()
os.chdir(og_dir)
os.chdir(og_dir + "/corpus_text")
lst = os.listdir()
print(lst)
for doc in lst:
    with open(doc) as json_file:
        data = json.load(json_file)
        print(len(data))
"""

create_corpus()