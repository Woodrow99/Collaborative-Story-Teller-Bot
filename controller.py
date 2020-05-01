"""
File Name: controller.py

Author: Drake Oswald

Description: This program establishes the controller for the TBA game 
for Intelligent Systems. The controller handles interactions between
the user and the agent that interprets sentences and creates scenerios.
Furthermore, the controller controlls the flow of data between the 
various aspects of the agent.
"""

import text_generation
from text_generation import generate_seq
import helper_functions
from helper_functions import load_game
from helper_functions import preprocess

model, cti, itc, chars = load_game("2LM")

def main():
    valid = True
    numOfTurns = ""

    while valid:
        numOfTurns = input("Insert a positive integer that represents how many turns that you have. ")

        while numOfTurns.isnumeric() == False:
            numOfTurns = input("Insert a positive integer that represents how many turns that you have. ")
        
        numOfTurns = int(numOfTurns)

        if numOfTurns >= 1:
            valid = False


    print("Enter some text to begin communicating.")

    for turn in range(numOfTurns):
        text = input("\n")
        #print((generate_seq(model, cti, itc, preprocess(text), chars, len(text), 1))[len(text):])
        print(generate_seq(model, cti, itc, preprocess(text), chars, len(text), 1))