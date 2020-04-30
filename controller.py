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

    numOfTurns = int(input("Insert a positive integer that represents how many turn that you want to play. "))
    print("Enter some text to begin communicating.")

    for turn in range(numOfTurns):
        text = input("")
        print((generate_seq(model, cti, itc, len(text), preprocess(text), chars, len(text), 1.0)) + "\n")
        

main()