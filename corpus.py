import os
import json
import nltk
import re
from nltk.corpus import PlaintextCorpusReader
from nltk import PorterStemmer
from nltk.corpus import stopwords 


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
blanks = ['', "\t", "\r", "\n", ' ']

def listToString(s):  
    str1 = ""  
    for ele in s:  
        str1 += ele + " "
    str1 = str1[:-1] + "\n"      
    return str1  

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
            if word not in blanks:
                word = word.lower()
                word = re.sub('\W+','', word)
                if word not in blanks:
                    normSentsList.append(word)



    start = 0
    end = 50
    sequenceList = []
    while end <= len(normSentsList):
        sequenceList.append(listToString(normSentsList[start:end]))
        start += 1 
        end += 1

    print(normSentsList[104131: 104181])
    
    """
    size = []

    for i in range(len(sequenceList)):
        if len(sequenceList[i]) != 50 and len(size) < 100:
            print(len(sequenceList[i]))
            size.append(i)

    print(sequenceList[0])
    print(size)
    """      

    fle = open("training_sentence.txt", "w")
    fle.writelines(sequenceList)


create_corpus()

 