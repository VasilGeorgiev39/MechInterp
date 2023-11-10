import plotly.express as px
import torch
import transformer_lens.utils as utils
device = "cuda"

def imshow(tensor, renderer=None, xaxis="", yaxis="", color_continuous_scale="RdBu", color_continuous_midpoint=0.0, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=color_continuous_midpoint, color_continuous_scale=color_continuous_scale, labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def imshowHTML(tensor, renderer=None, xaxis="", yaxis="", color_continuous_scale="RdBu", color_continuous_midpoint=0.0, **kwargs):
    return px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=color_continuous_midpoint, color_continuous_scale=color_continuous_scale, labels={"x":xaxis, "y":yaxis}, **kwargs).to_html(renderer)

def imshowImage(tensor, file, renderer=None, xaxis="", yaxis="", color_continuous_scale="RdBu", color_continuous_midpoint=0.0, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=color_continuous_midpoint, color_continuous_scale=color_continuous_scale, labels={"x":xaxis, "y":yaxis}, **kwargs).write_image(file=file, format="png")

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    return px.line(utils.to_numpy(tensor), **kwargs)#.show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def plot_square_as_board(state, diverging_scale=True, imgFile = None, html=False, **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    alpha = "ABCDEFGH"
    if imgFile:
        imshowImage(state, imgFile, x=[i for i in alpha], y=[str(i+1) for i in reversed(range(8))], color_continuous_scale="Blues", color_continuous_midpoint=None, aspect="equal", **kwargs)
    if html:
        return imshowHTML(state, x=[i for i in alpha], y=[str(i+1) for i in reversed(range(8))], color_continuous_scale="Blues", color_continuous_midpoint=None, aspect="equal", **kwargs)
    if diverging_scale:
        imshow(state, x=[i for i in alpha], y=[str(i+1) for i in reversed(range(8))], color_continuous_scale="RdBu", color_continuous_midpoint=0., aspect="equal", **kwargs)
    else:
        imshow(state, x=[i for i in alpha], y=[str(i+1) for i in reversed(range(8))], color_continuous_scale="Blues", color_continuous_midpoint=None, aspect="equal", **kwargs)

def visualisePredictions(logProbs, currentPos, dataProvider, cfg, html=False, img = False, moveNum = None):

    king_board_state = torch.zeros(64, device=device)
    bishop_board_state = torch.zeros(64, device=device)
    knight_board_state = torch.zeros(64, device=device)
    bking_board_state = torch.zeros(64, device=device)

    king_board_state -= 15.
    bishop_board_state -= 15.
    knight_board_state -= 15.
    bking_board_state -= 15.

    for i in range(cfg.d_vocab):
        moveAsString = dataProvider.decode(i)
        
        if moveAsString[0] == "K":
            board = king_board_state
        elif moveAsString[0] == "B":
            board = bishop_board_state
        elif moveAsString[0] == "N":
            board = knight_board_state
        elif moveAsString[0] == "k":
            board = bking_board_state
        else:
            continue

        columnString = moveAsString[1]
        columnIndex = ord(columnString) - ord('a')
        rowString = moveAsString[2]
        rowIndex = 8 - int(rowString)

        board[rowIndex * 8 + columnIndex] = logProbs[i]
    
    if img:
        plot_square_as_board(king_board_state.reshape(8, 8), imgFile=f"visualisations/game_log_probs/Move_{moveNum}_K.html", zmax=0, diverging_scale=False, title=f"Move {moveNum} King at pos {currentPos[1]} Log Probs", html=html)
        plot_square_as_board(bishop_board_state.reshape(8, 8), imgFile=f"visualisations/game_log_probs/Move_{moveNum}_B.html", zmax=0, diverging_scale=False, title=f"Move {moveNum} Bishop at pos {currentPos[0]} Log Probs", html=html)
        plot_square_as_board(knight_board_state.reshape(8, 8), imgFile=f"visualisations/game_log_probs/Move_{moveNum}_N.html", zmax=0, diverging_scale=False, title=f"Move {moveNum} Knight at pos {currentPos[2]} Log Probs", html=html)
        plot_square_as_board(bking_board_state.reshape(8, 8), imgFile=f"visualisations/game_log_probs/Move_{moveNum}_E.html", zmax=0, diverging_scale=False, title=f"Move {moveNum} Enemy at pos {currentPos[3]} Log Probs", html=html)
        return

    vis = []
    
    vis.append(plot_square_as_board(king_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title=f"Move {moveNum} King at pos {currentPos[1]} Log Probs", html=html))
    vis.append(plot_square_as_board(bishop_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title=f"Move {moveNum} Bishop at pos {currentPos[0]} Log Probs", html=html))
    vis.append(plot_square_as_board(knight_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title=f"Move {moveNum} Knight at pos {currentPos[2]} Log Probs", html=html))
    vis.append(plot_square_as_board(bking_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title=f"Move {moveNum} Enemy at pos {currentPos[3]} Log Probs", html=html))

    if html:
        return vis

def plotLogProbsForCurrentPosition(model, inputMoves, currentPos, dataProvider, cfg, html = False, img=False, moveNum=None):
    encodedMoves = [dataProvider.encode(move) for move in inputMoves]
    modelInput = torch.tensor([encodedMoves]).to(torch.int64).to("cuda")
    logits = model(input = modelInput, return_type = "logits")
    logit_vec = logits[0, -1]
    log_probs = logit_vec.log_softmax(-1)
    return visualisePredictions(log_probs, currentPos, dataProvider, cfg, moveNum=moveNum, html=html)

def plotLogProbsForEntireGame(model, game, dataProvider, cfg):
    gameSoFar = ['<BOS>']
    gameVis = []
    currentPos = ['f1', 'g1', 'h1','e5']
    actualMoveNum = 0
    for moveNum in range(len(game)):
        move = game[moveNum]
        if (move != '.'):
            if move[0]=='B':
                currentPos[0] = move[1:]
            elif move[0]=='K':
                currentPos[1] = move[1:]
            elif move[0]=='N':
                currentPos[2] = move[1:]
            elif move[0]=='k':
                currentPos[3] = move[1:]
            gameSoFar.append(move)
            paddedGame = ['.'] * (cfg.n_ctx - len(gameSoFar)) + gameSoFar
            gameVis.append(plotLogProbsForCurrentPosition(model, paddedGame, currentPos, dataProvider, cfg, moveNum=actualMoveNum, html=True))
            #plotLogProbsForCurrentPosition(model, paddedGame, img=True, moveIdx=moveIdx)
            actualMoveNum+=1
    return gameVis