"""
File Name: controller.py

Authors: Drake Oswald

Description: This program establishes the controller for the TBA game 
for Intelligent Systems. The controller handles interactions between
the user and the agent that interprets sentences and creates scenerios.
Furthermore, the controller controlls the flow of data between the 
various aspects of the agent.

Note: punkt and averaged_perceptron_tagger must be downloaded from ntk.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
#from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def game_controller(num_of_turns):
    """
    Input: An integer that represent the number of turns in a game.
    Furthermore, every interation of the game loop the user will input
    a string to standard input that represent the user's response to
    the given scenerio

    Output: Each iteration of the game loop the controller will present
    a scenerio to the user will respond to. The scenerio will be based
    on the user's response to the previous game loop.
    """

    intial_game_prompt = "Wine or Cheese? \n"
    user_responce = input(intial_game_prompt)
    while num_of_turns > 0:
        ffnn_input = preprocess(user_responce)
        print(ffnn_input)
        #feed ffnn_input into the FFNN
        #feed the output of the line above into the GCDNN
        if num_of_turns > 1:
            user_responce = input("insert scenerio here. \n")
        else:
            print("final scenerio")
        num_of_turns -= 1


def preprocess(sentence_str):
    """
    Input: A string that represents the response given by the user.

    Output: A list of pairs where the first item in the pair represents
    a normalized token, given as a string, from the input, and the
    second item in the pair contains a string representation of the 
    tag, the token type.  

    Note: preprocess currently maintains punctuation.
    """

    word_lst = word_tokenize(sentence_str)
    word_lst = [word.lower() for word in word_lst]
    word_lst = [word for word in word_lst if not word in stop_words]
    word_lst = [ps.stem(word) for word in word_lst]
    #word_tag_lst = nltk.pos_tag(word_lst)
    for word in word_lst:
        str_bytes(word)
    return word_lst


def str_bytes(word):
    binary = ""
    for char in word:
        temp = str(ord(char))
        if len(temp) < 3:
            temp = "0" + temp
        binary += temp
    binary = int(binary)
    b = bytes(decimalToBinary(binary), 'utf-8')
    print(b)


def decimalToBinary(n):  
    return bin(n).replace("0b", "")  