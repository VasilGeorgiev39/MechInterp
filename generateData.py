import json
import os
from random import randint
import chess
import chess.syzygy
from multiprocessing import Pool 

current_dir = os.path.dirname(__file__)
TABLE_LOCATION = os.path.join(current_dir, "endings_table")

gamesPerThread = 2000
generateStrongBlackMoves = True
def generateMoves(id):
    games = []
    with chess.syzygy.open_tablebase(TABLE_LOCATION) as tablebase:
        print("start")
        for t in range(gamesPerThread):
            board = chess.Board("8/8/8/4k3/8/8/8/5BKN w - - 0 1")
            i = 0
            finalMoves = []
            while True:
                i+=1
            
                moves = list(board.legal_moves)

                if (moves == []):
                    break

                if i%2 == 0:
                    if generateStrongBlackMoves:
                        dtz = []
                        for move in moves:
                            board.push(move)
                            dtz.append(tablebase.probe_dtz(board))
                            board.pop()
                        movesAndDtz = zip(moves, dtz)
                        sortedMoves = sorted(movesAndDtz, key=lambda x: x[1])
                        nextMove = sortedMoves[0][0]
                    else:
                        nextMove = moves[randint(0, len(moves) - 1)]
                else:
                    dtz = []
                    for move in moves:
                        board.push(move)
                        dtz.append(tablebase.probe_dtz(board))
                        board.pop()
                    dtz = [-x for x in dtz]
                    movesAndDtz = zip(moves, dtz)

                    noBadMoves = filter(lambda x: x[1] > 0, movesAndDtz)

                    sortedMoves = sorted(noBadMoves, key=lambda x: x[1])
                    #print(sortedMoves)

                    if i < 20:
                        moveIdx = randint(0, min(6, len(sortedMoves)) - 1)
                        nextMove = sortedMoves[moveIdx][0]
                    else:
                        nextMove = sortedMoves[0][0]
                moveStr = board.san_and_push(nextMove)
                moveStr = moveStr[:3]
                if (i % 2 == 0):
                    moveStr = moveStr.lower()
                finalMoves.append(moveStr)
                
            if id == 0 and t%100 == 0:
                print(t)
            games.append(finalMoves)
    
    with open(TABLE_LOCATION + "\\games" + str(id) + ".txt", "w") as f:
        json.dump(games, f)

if __name__ == '__main__':

    numProcesses = 8
    pool = Pool(processes=numProcesses)

    pool.map(generateMoves, range(numProcesses))

    allGames = []

    for i in range(numProcesses):
        fileName = TABLE_LOCATION + "\\games" + str(i) + ".txt"
        with open(fileName, "r") as f:
            games = json.load(f)
            allGames += games
        os.remove(fileName)

    with open(TABLE_LOCATION + "\\games.txt", "w") as f:
        json.dump(allGames, f)

    print("done")


