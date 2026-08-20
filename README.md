# LSTM From Scratch — NumPy Implementation + Mechanistic Interpretability

A character-level LSTM implemented entirely from scratch in NumPy — forward pass, manual backpropagation through time across all four gates, AdaGrad training — with no autograd anywhere. Trained on tinyshakespeare, then probed to find out what individual neurons inside it actually learned.

## What this is

Most from-scratch RNN/LSTM implementations stop at "it trains and the loss goes down." This project goes one step further: after training, it asks what individual hidden units are actually doing, and finds two neurons with genuinely interpretable, non-obvious behavior — not injected or expected, just found by looking.

## How it was built

**The model** (`lstm.py` / Sections 1–5 of the notebook): standard LSTM gate equations, all derived and implemented by hand:

```
f_t = σ(Wf · [h_{t-1}, x_t] + bf)      forget gate
i_t = σ(Wi · [h_{t-1}, x_t] + bi)      input gate
g_t = tanh(Wc · [h_{t-1}, x_t] + bc)   cell candidate
o_t = σ(Wo · [h_{t-1}, x_t] + bo)      output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        cell state (additive update)
h_t = o_t ⊙ tanh(c_t)
```

The backward pass is manual truncated BPTT — every gradient through every gate derived by hand, including the cell-state highway term (`dcnext = f_t * dc`) that is the actual mechanism by which LSTMs avoid the vanishing-gradient problem vanilla RNNs have: gradient flows backward through multiplication by a forget gate close to 1, instead of being squashed by a `tanh` derivative at every single step.

Trained on tinyshakespeare with AdaGrad, sequence length 100, hidden size 128.

## The experiment: finding interpretable neurons

Once trained, the model was run character-by-character over a probe passage (Shakespeare dialogue), recording the **forget gate value** for every one of the 128 neurons at every character position. That gives a `(num_characters, 128)` matrix — plotted as a heatmap, neurons on one axis, characters on the other, brightness = forget gate value (white = remembering, black = forgetting).

Scanning that full 128-neuron heatmap by eye — not searching for anything specific, just looking for any neuron whose column looked visually different from the noisy majority — surfaced two standout neurons.

## What was found

### Neuron 3 — tracks the word "citizen"

Neuron 3's forget gate stayed high (remembering) specifically during occurrences of the word **"citizen"**, and dropped to near-zero (forgetting) almost everywhere else in the passage. Every other word, character, and pause in the text left this neuron essentially flat — it wasn't responding to punctuation, sentence structure, or generic word boundaries, just this one recurring token. Zooming into neuron 3's row alone and overlaying the actual characters on the heatmap confirmed the pattern holds consistently across every occurrence of "citizen" in the probe text, not just one lucky instance.

### Neuron 78 — resets after "."

Neuron 78 showed the opposite kind of pattern: its forget gate would **drop sharply right after a period**, consistently, across the passage — effectively wiping its own contribution to the cell state every time a sentence ended. Everywhere else, it held a relatively stable value. This looks like the network independently learning a sentence-boundary reset mechanism — using the forget gate exactly the way it's designed to be used, to clear out state that's no longer relevant once a new sentence starts.

## Why this matters

Neither of these behaviors was hand-designed or expected in advance — the model was trained purely on a next-character prediction loss with no supervision about words, sentences, or entity tracking. Both patterns emerged on their own and were only found afterward, by inspecting the trained weights. That's the basic move of mechanistic interpretability: don't just trust that a black-box model works, open it up and check what's actually happening inside, one neuron at a time.

## Files

- `LSTM_from_scratch.ipynb` — the full pipeline: data loading → model → training → interpretability heatmaps → the neuron 3 / neuron 78 findings, in one notebook, in order.

## Running it

Open the notebook and run top to bottom. Training runs for 100,000 iterations at hidden size 128 (takes a while on CPU — this matches the original run). Once trained, the interpretability cells at the end reproduce the heatmaps and the neuron 3 / neuron 78 findings above.
