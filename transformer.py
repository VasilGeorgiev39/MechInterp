import copy
import pickle
import webbrowser
import einops
import numpy as np
import torch
import tqdm

from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.utils as utils

import dataProcessor as dp

from jaxtyping import Float
from typing import Callable

from torch import Tensor

from functools import partial

import circuitsvis as cv

import neel_plotly

from sampler import *

from utils import *

import os

current_dir = os.path.dirname(__file__)

# where we save the model
PTH_LOCATION = os.path.join(current_dir, "model\\chessgpt.pth")

cfg = HookedTransformerConfig(
    n_layers = 4,
    d_head = 64,
    n_heads = 4,
    d_model = 64 * 4,
    d_mlp = 4 * (64 * 4),
    d_vocab = 221,
    n_ctx = 100,
    act_fn="gelu",
    normalization_type="LNPre",
    init_weights=True
)
device = "cuda"

# Training parameters
batchSize = 512
lr = 1e-5
wd = 0.01 
betas = (0.9, 0.98)

num_epochs = 500000
checkpoint_every = 1000

TRAIN_MODEL = False
LOAD_MODEL = True

train_losses = []
test_losses = []
model_checkpoints = []
checkpoint_epochs = []

# Unused - currently using standard cross entropy loss
def loss_fn(logits, labels):
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)

    correct_log_probs = torch.zeros_like(labels).to(torch.float64)
    for batchNum in range(log_probs.shape[0]):
        for seqNum in range(log_probs.shape[1] - 1):
            correct_log_probs[batchNum][seqNum] = log_probs[batchNum][seqNum][labels[batchNum][seqNum]]
            
    #correct_log_probs = log_probs.gather(dim=-1, index=labels[:, :, None])[:, :, 0]
    #print(correct_log_probs)
    return -correct_log_probs.mean()

dataProvider = dp.DataPovider(cfg.n_ctx)

model = None

if LOAD_MODEL:
    model = HookedTransformer(cfg)
    cached_data = torch.load(PTH_LOCATION)
    model.load_state_dict(cached_data['model'])
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]
    test_losses = cached_data['test_losses']
    train_losses = cached_data['train_losses']
    vocab = cached_data["vocab"]
    dataProvider.loadVocab(vocab)

    #neel_plotly.line_or_scatter(train_losses, plot_type = "line", title="Training loss", line_labels=["loss"], width=750)

#playHuman(model, dataProvider)
#getStats(model, dataProvider)
#exit()

if TRAIN_MODEL:

    dataProvider.loadData()

    cfg.d_vocab = len(dataProvider.vocab)
    cfg.d_vocab_out = len(dataProvider.vocab)
    if not model:
        model = HookedTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_data = dataProvider.getBatch(True, batchSize)

        train_logits, train_loss = model(input = train_data, return_type = "both")
        #train_loss = loss_fn(train_logits, train_labels)
        #print(loss, train_loss)
        train_loss.backward()
        train_losses.append(train_loss.item())

        optimizer.step()
        optimizer.zero_grad()
        
        if ((epoch+1)%checkpoint_every)==0:
            with torch.inference_mode():
                test_data = dataProvider.getBatch(False, batchSize)
                test_logits, test_loss = model(test_data, return_type = "both")
                #test_loss = loss_fn(test_logits, test_labels)
                test_losses.append(test_loss.item())
            checkpoint_epochs.append(epoch)
            model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")

            torch.save(
                {
                    "model":model.state_dict(),
                    "config": model.cfg,
                    "checkpoints": model_checkpoints,
                    "checkpoint_epochs": checkpoint_epochs,
                    "test_losses": test_losses,
                    "train_losses": train_losses,
                    "vocab": dataProvider.vocab,
                    #"train_indices": train_indices,
                    #"test_indices": test_indices,
                },
                PTH_LOCATION)

# Exploration

torch.enable_grad(False)

#https://lichess.org/study/ruqNokDo/nnb9HfBr
#position just before mate move - Bc6
inputMoves = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '<BOS>', 'Ba6', 'kd6', 'Kg2', 'kc5', 'Kg3', 'kb6', 'Bd3', 'kc5', 'Kg4', 'kd4', 'Ba6', 'kc5', 'Kf5', 'kb6', 'Be2', 'kc5', 'Ke5', 'kb6', 'Bc4', 'kc5', 'Bf7', 'kb6', 'Ng3', 'kc5', 'Nf5', 'kb6', 'Kd6', 'kb7', 'Be8', 'kc8', 'Bc6', 'kd8', 'Nd4', 'kc8', 'Nb3', 'kd8', 'Nc5', 'kc8', 'Bd7', 'kb8', 'Kc6', 'ka7', 'Kc7', 'ka8', 'Kb6', 'kb8', 'Na6', 'ka8']

#inputMoves = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '<BOS>', 'Ba6', 'kd6', 'Kg2', 'kc5', 'Kg3', 'kb6', 'Bd3', 'kc5', 'Kg4', 'kd4', 'Ba6', 'kc5', 'Kf5', 'kb6', 'Be2', 'kc5', 'Ke5', 'kb6', 'Bc4', 'kc5', 'Bf7', 'kb6', 'Ng3', 'kc5', 'Nf5']
#print(len(inputMoves))

incorrectMoveString = "Bb5"
incorrectMoveEncoded = dataProvider.encode(incorrectMoveString)

encodedMoves = [dataProvider.encode(move) for move in inputMoves]

modelInput = torch.tensor([encodedMoves]).to(torch.int64).to(device)

# Plot Log Probs for Entrire game
""" gameVis = plotLogProbsForEntireGame(model, inputMoves, dataProvider, cfg)

for moveIdx in range(len(gameVis)):
    moveVis = gameVis[moveIdx]
    with open(f"visualisations/game_log_probs/Move_{moveIdx}_K.html", "wb") as f:
        pickle.dump(str(gameVis[moveIdx][0]), f)
    with open(f"visualisations/game_log_probs/Move_{moveIdx}_B.html", "wb") as f:
        pickle.dump(str(gameVis[moveIdx][1]), f)
    with open(f"visualisations/game_log_probs/Move_{moveIdx}_N.html", "wb") as f:
        pickle.dump(str(gameVis[moveIdx][2]), f)
    with open(f"visualisations/game_log_probs/Move_{moveIdx}_E.html", "wb") as f:
        pickle.dump(str(gameVis[moveIdx][3]), f) """

logits, cache = model.run_with_cache(input = modelInput, return_type = "logits")
logit_vec = logits[0, -1]
log_probs = logit_vec.log_softmax(-1)

predictedMoveEmbeded = torch.argmax(logit_vec)
predictedMoveString = dataProvider.decode(predictedMoveEmbeded.item())
print("Predicted move: ", predictedMoveString)

#visualisePredictions(log_probs)

# Logit attribution

accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
# accumulated_residual has shape (component, batch, d_model)

correct_tokens = modelInput[:, -1]
incorrect_tokens = torch.zeros_like(correct_tokens)
incorrect_tokens[0] = incorrectMoveEncoded
answer_tokens = torch.stack((correct_tokens, incorrect_tokens), dim=1)
#print("Answer tokens shape:", answer_tokens.shape)

answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
#print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
#print(f"Logit difference directions shape:", logit_diff_directions.shape)

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
) -> Float[Tensor, "..."]:
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size

logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache)

fig = line(
    logit_lens_logit_diffs,
    title="Logit Difference From Accumulated Residual Stream",
    xaxis="Layer",
    yaxis="Logit Diff",
    width=800
)

fig.update_layout(xaxis_tickvals=list(range(len(labels))), xaxis_ticktext=labels)
fig.show()

per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)

fig = line(
    per_layer_logit_diffs,
    title="Logit Difference From Each Layer",
    xaxis="Layer",
    yaxis="Logit Diff",
    width=800
)

fig.update_layout(xaxis_tickvals=list(range(len(labels))), xaxis_ticktext=labels)
fig.show()


per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual,
    "(layer head) ... -> layer head ...",
    layer=model.cfg.n_layers
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)

imshow(
    per_head_logit_diffs,
    xaxis="Head",
    yaxis="Layer",
    title="Logit Difference From Each Head",
    width=600
)

# Attention patterns

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()


k = 4

for head_type in ["Positive", "Negative"]:

    # Get the heads with largest (or smallest) contribution to the logit difference
    top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)

    # Get all their attention patterns
    attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack([
        cache["pattern", layer][:, head][0]
         for layer, head in top_heads
    ])

    attn_heads = cv.attention.attention_heads(
        attention = attn_patterns_for_important_heads,
        tokens = inputMoves,
        attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
    )

    path = "attn_heads_" + head_type + ".html"

    with open(path, "w") as f:
        f.write(str(attn_heads))

    webbrowser.open(path)

# Activation patching

clean_input = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '<BOS>', 'Kg2', 'kd4', 'Kh3', 'ke4', 'Bg2', 'kf4', 'Bb7', 'ke5', 'Kg3', 'kf5', 'Kf3', 'ke5', 'Ng3', 'kd6', 'Be4', 'kc5', 'Nf5', 'kc4', 'Ke3', 'kc5', 'Kd2', 'kc4', 'Bb7', 'kc5', 'Kc3', 'kb6', 'Bd5', 'ka5', 'Ng7', 'kb6', 'Ne6', 'ka5', 'Bc6', 'kb6', 'Be8', 'ka6', 'Kb4', 'kb7']
""" In this position the model predicts Kb5
    I'm trying to patch it to think that black moved Kb6 as last move which makes Kb5 illegal
"""

corrupted_input = clean_input.copy()
corrupted_input[-1] = 'kb6'

cleanEncodedMoves = [dataProvider.encode(move) for move in clean_input]
corruptedEncodedMoves = [dataProvider.encode(move) for move in corrupted_input]

cleanMoveIndex = cleanEncodedMoves[-1]
corruptedMoveIndex = corruptedEncodedMoves[-1]
correctMoveString = 'Kb5'
correctMoveIndex = dataProvider.encode(correctMoveString)

cleanInputTensor = torch.tensor([cleanEncodedMoves]).to(torch.int64).to("cuda")
corruptedInputTensor = torch.tensor([corruptedEncodedMoves]).to(torch.int64).to("cuda")

clean_logits, cleanCache = model.run_with_cache(input = cleanInputTensor)

predictedMoveEmbeded = torch.argmax(clean_logits[0][-1])
predictedMoveString = dataProvider.decode(predictedMoveEmbeded.item())
#print("Predicted move: ", predictedMoveString)

corrupted_logits, corruptedCache = model.run_with_cache(input = corruptedInputTensor)

clean_log_probs = clean_logits.log_softmax(dim=-1)
corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)

clean_Kb5_log_prob = clean_log_probs[0, -1, correctMoveIndex]
corrupted_Kb5_log_prob = corrupted_log_probs[0, -1, correctMoveIndex]

#print("Clean Kb5 log prob", clean_Kb5_log_prob.item())
#print("Corrupted Kb5 log prob", corrupted_Kb5_log_prob.item(), "\n")

def patching_metric(patched_logits: Float[Tensor, "batch=1 seq d_vocab"]):
    patched_log_probs = patched_logits.log_softmax(dim=-1)
    return (patched_log_probs[0, -1, correctMoveIndex] - corrupted_Kb5_log_prob) / (clean_Kb5_log_prob - corrupted_Kb5_log_prob)

def patch_final_move_output(
    activation: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook function which patches activations at the final sequence position.

    Note, we only need to patch in the final sequence position, because the
    prior moves in the clean and corrupted input are identical (and this is
    an autoregressive model).
    '''

    activation[0, -1, :] = clean_cache[hook.name][0, -1, :]
    return activation

def get_act_patch_resid_pre(
    model: HookedTransformer,
    corrupted_input: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[Float[Tensor, "batch seq d_model"]], Float[Tensor, ""]]
) -> Float[Tensor, "2 n_layers"]:
    '''
    Returns an array of results, corresponding to the results of patching at
    each (attn_out, mlp_out) for all layers in the model.
    '''
    
    model.reset_hooks()
    results = torch.zeros(2, model.cfg.n_layers, device=device, dtype=torch.float32)
    hook_fn = partial(patch_final_move_output, clean_cache=clean_cache)

    for i, activation in enumerate(["attn_out", "mlp_out"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            patched_logits = model.run_with_hooks(
                corrupted_input,
                fwd_hooks = [(utils.get_act_name(activation, layer), hook_fn)],
            )
            results[i, layer] = patching_metric(patched_logits)

    return results

patching_results = get_act_patch_resid_pre(model, corruptedInputTensor, cleanCache, patching_metric)

neel_plotly.line_or_scatter(patching_results, plot_type = "line", title="Layer Output Patching Effect on Kb5 Log Prob", line_labels=["attn", "mlp"], width=750)





