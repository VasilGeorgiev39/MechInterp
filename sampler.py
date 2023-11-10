from random import randint
import chess
import chess.syzygy
import torch
from tqdm import tqdm
import os

current_dir = os.path.dirname(__file__)
TABLE_LOCATION = os.path.join(current_dir, "endings_table")

def playHuman(model, dataProvider):

    board = chess.Board("8/8/8/4k3/8/8/8/5BKN w - - 0 1")
    
    with torch.inference_mode():
        inputMoves = dataProvider.getStartInput()
        encodedMoves = [dataProvider.encode(move) for move in inputMoves]

        modelInput = torch.tensor([encodedMoves]).to(torch.int64).to("cuda")
        
        while True:
            sample_logits = model(input = modelInput, return_type = "logits")
            #print(sample_logits.shape)
            #print(sample_logits[0][-1])
            bestMoveEncoded = torch.argmax(sample_logits[0][-1]).item()
            bestMove = dataProvider.decode(bestMoveEncoded)
            
            encodedMoves = encodedMoves[1:] + [bestMoveEncoded]
            decodedMoves = [dataProvider.decode(x) for x in encodedMoves]
            print(decodedMoves)
            board.push_san(bestMove)
            print(board)

            humanMove = input("Enter your move: ")
            if (humanMove == "quit"):
                break
            humanMoveEncoded = dataProvider.encode(humanMove)
            encodedMoves = encodedMoves[1:] + [humanMoveEncoded]
            modelInput = torch.tensor([encodedMoves]).to(torch.int64).to("cuda")
            board.push_san(humanMove.capitalize())

def playBoard(model, dataProvider, bestDefence=False):
    board = chess.Board("8/8/8/4k3/8/8/8/5BKN w - - 0 1")
    validMoves = 0
    with torch.inference_mode():
        with chess.syzygy.open_tablebase(TABLE_LOCATION) as tablebase:
            inputMoves = dataProvider.getStartInput()
            encodedMoves = [dataProvider.encode(move) for move in inputMoves]

            modelInput = torch.tensor([encodedMoves]).to(torch.int64).to("cuda")
            
            while True:
                #print(modelInput)
                #decodedMoves = [dataProvider.decode(x.item()) for x in modelInput[0]]
                #print(decodedMoves)
                sample_logits = model(modelInput)
                bestMoveEncoded = torch.argmax(sample_logits[0][-1]).item()
                bestMove = dataProvider.decode(bestMoveEncoded)
                try:
                    board.push_san(bestMove)
                except:
                    #print(board)
                    #print(bestMove)
                    return [False, validMoves, 1]
                
                encodedMoves = encodedMoves[1:] + [bestMoveEncoded]
                validMoves += 1
                
                if board.is_checkmate():
                    return [True, validMoves, 0]
                
                if board.is_stalemate():
                    return [False, validMoves, 0]
                
                if validMoves == 50:
                    return [False, validMoves, 0]
                
                moves = list(board.legal_moves)
                if bestDefence:
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
                nextMoveStr = board.san(nextMove)
                nextMoveStr = nextMoveStr[:3]
                nextMoveStr = nextMoveStr.lower()
                if (nextMoveStr[1] == 'x'):
                    return [False, validMoves, 0]
                nextMoveEncoded = dataProvider.encode(nextMoveStr)
                encodedMoves = encodedMoves[1:] + [nextMoveEncoded]
                modelInput = torch.tensor([encodedMoves]).to(torch.int64).to("cuda")
                board.push(nextMove)

def getStats(model, dataProvider, bestDefence=True):
    totalInvalidMoves = 0
    totalValidMoves = 0
    successfulGames = 0
    t = 100
    for i in tqdm(range(t)):
        success, validMoves, invalidMoves = playBoard(model, dataProvider)
        if success:
            successfulGames += 1
        totalValidMoves += validMoves
        totalInvalidMoves += invalidMoves
    successfulGamesPecent = float(successfulGames) / t
    validMovesPercent = float(totalValidMoves) / (totalValidMoves + totalInvalidMoves)

    print("Successful games: " + str(successfulGames))
    print(f"Successful games %: {successfulGamesPecent:,.2%}")
    print("Invalid moves: " + str(totalInvalidMoves))
    print("Valid moves: " + str(totalValidMoves))
    print(f"Valid moves %: {validMovesPercent:,.2%}")