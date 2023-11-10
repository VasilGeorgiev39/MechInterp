import json
import chess
from matplotlib import pyplot as plt

import numpy as np
import torch
from random import shuffle
from tqdm import tqdm

import os

current_dir = os.path.dirname(__file__)
DATASET_LOCATION = os.path.join(current_dir, "games.txt")

class DataPovider:
    def __init__(self, seqLength) -> None:
        self.games = []
        self.vocab = None
        self.stoi = None
        self.itos = None
        self.seqLength = seqLength
    
    def loadData(self, generateStats = False):
        print("Reading file...")
        with open(DATASET_LOCATION, "r") as f:
            data = json.load(f)
        print("Parsing data...")
        moveFrequency = {}
        allGamesList = [x for batch in data for x in batch]

        paddedGames = []

        for game in tqdm(allGamesList):
            paddedGame = self.getStartInput()
            for move in game:
                if move not in moveFrequency:
                    moveFrequency[move] = 0
                moveFrequency[move] += 1
                paddedGame = paddedGame[1:] + [move]
                paddedGames.append(paddedGame)

        if generateStats:
            print("Total number of games: " + str(len(allGamesList)))

            gamesAsSingleString = [''.join(x) for x in allGamesList]
            uniqueGames = set(gamesAsSingleString)
            print("Number of unique games: " + str(len(uniqueGames)))

            uniqueBoards = set()
            for game in tqdm(allGamesList):
                board = chess.Board("8/8/8/4k3/8/8/8/5BKN w - - 0 1")
                for move in game:
                    board.push_san(move.capitalize())
                    uniqueBoards.add(board.fen())

            print("Number of unique positions: " + str(len(uniqueBoards)))

            print("Total number of moves: " + str(sum([len(x) for x in allGamesList])))
            print("Longest game: " + str(max([len(x) for x in allGamesList])))
            print("Shortest game: " + str(min([len(x) for x in allGamesList])))
            print("Average game length: " + str(sum([len(x) for x in allGamesList]) / len(allGamesList)))
            print("Number of unique moves: " + str(len(moveFrequency)))
            sortedMoves = sorted(moveFrequency.items(), key=lambda x: x[1])
            print("Most common moves: " + str(sortedMoves[-10:]))
            print("Least common moves: " + str(sortedMoves[:10]))

            self.plotMoveFrequencyBarChart(moveFrequency)

        self.vocab = ['.'] + ["<BOS>"] + [x[0] for x in tqdm(moveFrequency.items())]

        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }

        for game in tqdm(paddedGames):
            self.games.append([self.encode(move) for move in game])

        shuffle(self.games)
        
    def encode(self, move):
        return self.stoi[move]
    
    def decode(self, move):
        return self.itos[move]

    def getBatch(self, train, batchSize):
        
        if train:
            minIdx = 0
            maxIdx = int(len(self.games) * 0.8)
        else:
            minIdx = int(len(self.games) * 0.8)
            maxIdx = len(self.games)

        idxs = np.random.randint(minIdx, maxIdx, size=batchSize)

        gamesToReturn = [self.games[x] for x in idxs]
        return torch.tensor(gamesToReturn)
    
    def getStartInput(self):
        return ["."] * (self.seqLength-1) + ["<BOS>"]

    def getVocab(self):
        return self.vocab
    
    def loadVocab(self, vocab):
        self.vocab = vocab
        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }

    def isGameInSet(self, gameToCheck):
        for g in self.games:
            if g == gameToCheck:
                return True
        return False
    
    def plotMoveFrequencyBarChart(self, moveFrequency):
        sortedMoves = sorted(moveFrequency.items(), key=lambda x: x[1], reverse=True)
        sortedKeys = [x[0] for x in sortedMoves]
        sortedValues = [x[1] for x in sortedMoves]
        #print(sortedMoves[:10])
        #print(x)
        #print(y)
        plt.bar(sortedKeys,sortedValues)
        plt.show()

if __name__ == "__main__":
    dp = DataPovider(100)
    dp.loadData(generateStats=True)

    testGame = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '<BOS>', 'Kg2', 'kd4', 'Kh3', 'ke4', 'Bg2', 'kf4', 'Bb7', 'ke5', 'Kg3', 'kf5', 'Kf3', 'ke5', 'Ng3', 'kd6', 'Be4', 'kc5', 'Nf5', 'kc4', 'Ke3', 'kc5', 'Kd2', 'kc4', 'Bb7', 'kc5', 'Kc3', 'kb6', 'Bd5', 'ka5', 'Ng7', 'kb6', 'Ne6', 'ka5', 'Bc6', 'kb6', 'Be8', 'ka6', 'Kb4', 'kb7']
    print(dp.isGameInSet(testGame))
    games = dp.getBatch(True, 10)

    decodedGame = [dp.decode(x.item()) for x in games[0]]
    print(decodedGame)





