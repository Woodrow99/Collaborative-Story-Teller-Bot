"""
File Name: controller.py

Author: Drake Oswald

Description: This program establishes the controller for the system.
The controller handles interactions between the user and the model 
that interprets sentences and generates text.
"""

import text_generation
from text_generation import generate_seq
import helper_functions
from helper_functions import load_game
from helper_functions import preprocess

model, cti, itc, chars = load_game("2LM")

def main():
    numOfTurns = ""

    while True:
        numOfTurns = input("Insert a positive integer that represents how many turns that you want. ")

        while not numOfTurns.isnumeric():
            numOfTurns = input("Insert a positive integer that represents how many turns that you want. ")
        
        numOfTurns = int(numOfTurns)

        if numOfTurns >= 1:
            break

    while True:
        temp = input("Insert a positive number between .2 and 1.5 that represents the temperature. \n" \
        "The temperature represents how conservative or creative the model generates charcters. ")

        while not is_float(temp):
            temp = input("Insert a positive number between .2 and 1.5 that represents the temperature. \n" \
            "The temperature represents how conservative or creative the model generates charcters. ")
        
        temp = float(temp)

        if temp <= 1.5 and temp >= .2:
            break

    print("Enter some text to begin communicating.")

    for x in range(numOfTurns):
        text = input("\n")
        print("\n"+(generate_seq(model, cti, itc, preprocess(text), chars, len(text), temp))[len(text):])
        #print(generate_seq(model, cti, itc, preprocess(text), chars, len(text), 1))

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

main()