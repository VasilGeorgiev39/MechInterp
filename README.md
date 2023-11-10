# Intro

This is my application projects for the [MATS](https://www.matsprogram.org/) program

The code is extremely ugly and hacky, please don't judge me for it lol. I have prioritized speed and making things working quickly over good software engineering.

I have liberaly copied code from [Neel Nanda](https://github.com/neelnanda-io)'s tutorials, [ARENA](https://www.arena.education/), Google, Stackoverflow and ChatGPT but all mistakes are mine obviously.

# Project

I trained a small (3 million parameters) GPT-2 style model to play bishop-and-knight chess endgames and I tried to use mechanistic interpretability techniques to understand how it makes decisions and look for internal representations of the game state.

The model was producing a legal move in 99% of the cases
The model was reaching a checkmate in 72% of the games

# Report

For more information check the [report](https://github.com/VasilGeorgiev39/MechInterp/blob/main/Absent%20World%20Representations.md). (there is also an [html version](https://github.com/VasilGeorgiev39/MechInterp/blob/main/Absent%20World%20Representations.html) that has a nice script to view the log probabilities of all moves in a given game but it is very heavy to load)

