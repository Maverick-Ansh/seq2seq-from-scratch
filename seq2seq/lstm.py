"""
seq2seq.lstm — the LSTM, written from its equations.

WHY THIS FILE EXISTS
--------------------
Sutskever, Vinyals & Le (2014) is a paper about *what you can build out of an
LSTM*, not about the LSTM itself. But you cannot understand the paper if the
LSTM is a black box, so we build one here from the four equations, and then we
prove — numerically, to ~1e-6 — that ours is the same object as PyTorch's
`nn.LSTM`. After that proof you are allowed to use the fast cuDNN version and
still claim you know what it does.

THE FOUR EQUATIONS
------------------
An LSTM carries TWO pieces of state from step t-1 to step t:

    c   the "cell state"   — the long-term memory, a conveyor belt
    h   the "hidden state" — the short-term / output view of that memory

At each timestep it looks at the new input x_t and the previous h_{t-1}, and
computes four vectors of the same width H:

    i_t = sigmoid(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)     INPUT gate
    f_t = sigmoid(W_if x_t + b_if + W_hf h_{t-1} + b_hf)     FORGET gate
    g_t = tanh   (W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)     CANDIDATE
    o_t = sigmoid(W_io x_t + b_io + W_ho h_{t-1} + b_ho)     OUTPUT gate

    c_t = f_t * c_{t-1}  +  i_t * g_t
    h_t = o_t * tanh(c_t)

Read it as English:

  * g_t is "what I would like to write into memory this step" (tanh, so it is
    signed: it can push a feature up or down).
  * i_t in [0,1] is "how much of that am I actually allowed to write".
  * f_t in [0,1] is "how much of the old memory do I keep". f=1 means keep
    everything, f=0 means wipe it.
  * o_t in [0,1] is "how much of my memory do I expose to the outside world
    this step". Memory can be held privately and released later.

WHY THIS BEATS A PLAIN RNN (the whole point)
--------------------------------------------
A vanilla RNN does h_t = tanh(W h_{t-1} + U x_t). To get the gradient from step
T back to step 1 you multiply T Jacobians together, each one roughly W^T times
a tanh' factor that is <= 1. That product behaves like W^T: if the largest
singular value of W is < 1 the gradient decays geometrically to zero (it
*vanishes*), if > 1 it explodes. Either way, a plain RNN cannot learn a
dependency 30 steps long.

The LSTM's escape hatch is the c_t line: dc_t/dc_{t-1} = f_t, an ELEMENTWISE
multiply, with no weight matrix in the path. If the network learns f ~ 1 for
some coordinate, the gradient flows back through that coordinate essentially
undamped, for hundreds of steps. That undamped highway is called the "constant
error carousel", and it is the only reason a 4-layer LSTM can read a
30-word sentence and still remember its first word.

We demonstrate exactly this, with numbers, in `grad_flow_demo` below.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# A (h, c) pair for one layer. PyTorch calls this the "hidden state" even
# though it is really two tensors; we keep the same convention so our modules
# are drop-in compatible with nn.LSTM.
State = Tuple[torch.Tensor, torch.Tensor]


class LSTMCellScratch(nn.Module):
    """One LSTM timestep, computed by hand.

    Shapes:
        x : (B, input_size)      one timestep of a batch of B sequences
        h : (B, hidden_size)
        c : (B, hidden_size)

    IMPLEMENTATION NOTE — why the weights are stored as ONE fat matrix.
    The four gates each need their own W_x* and W_h*, so naively that is eight
    matrix multiplies per timestep. Instead we stack the four weight matrices
    vertically into a single (4H, in) matrix and do ONE matmul, then slice the
    result into four chunks. Mathematically identical, but a single big GEMM
    is far friendlier to a GPU than four small ones — this is the same trick
    cuDNN uses, and it is why we can later copy weights back and forth with
    `nn.LSTM` with no reshaping.

    The chunk order is PyTorch's: [input, forget, candidate, output] = i,f,g,o.
    We follow it exactly so that weight-copying is a no-op.
    """

    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # (4H, in) and (4H, H) — the ".T" happens inside F.linear, so a row of
        # this matrix is one output unit's weights.
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        # PyTorch keeps TWO bias vectors (b_ih and b_hh) even though their sum
        # is the only thing that matters. It is a historical artifact of the
        # cuDNN API. We mirror it so state_dicts are interchangeable.
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        self.forget_bias = forget_bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """PyTorch's default LSTM init: U(-1/sqrt(H), +1/sqrt(H)).

        (The *paper* uses U(-0.08, 0.08) instead — see `paper_init` in
        seq2seq/model.py. We keep PyTorch's default here so that the
        equivalence test starts from a realistic weight scale.)
        """
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)
        if self.forget_bias != 0.0:
            # Optional trick (Gers 1999, popularised by Jozefowicz 2015): start
            # the forget gate biased OPEN so that f≈1 at initialisation and the
            # memory highway is wide open before any learning happens. The 2014
            # paper does not do this; we expose it so you can A/B it yourself.
            with torch.no_grad():
                H = self.hidden_size
                self.bias_ih[H:2 * H].fill_(self.forget_bias)

    def forward(self, x: torch.Tensor, state: State) -> State:
        h_prev, c_prev = state

        # ONE matmul for all four gates of both the input and recurrent paths.
        # gates: (B, 4H)
        gates = (
            x @ self.weight_ih.t() + self.bias_ih
            + h_prev @ self.weight_hh.t() + self.bias_hh
        )

        # Split the fat vector back into the four gate pre-activations.
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)   # write gate      in [0, 1]
        f = torch.sigmoid(f)   # keep gate       in [0, 1]
        g = torch.tanh(g)      # candidate       in [-1, 1]
        o = torch.sigmoid(o)   # expose gate     in [0, 1]

        # The two lines that are the entire reason LSTMs work.
        c = f * c_prev + i * g          # <-- gradient highway: dc/dc_prev = f
        h = o * torch.tanh(c)

        return h, c


class LSTMLayerScratch(nn.Module):
    """Run an `LSTMCellScratch` across time.

    Input  : (T, B, input_size)   — TIME FIRST. See note below.
    Output : (T, B, hidden_size), final (h, c)

    WHY TIME-FIRST (T, B, H) AND NOT (B, T, H)?
    Because the recurrence is a Python loop over T, and `x[t]` on a time-first
    tensor is a contiguous slice — no copy, no stride games. It is also what
    nn.LSTM defaults to (`batch_first=False`). Every tensor in this repo is
    time-first unless its name says otherwise; when you see a shape you don't
    recognise, that is the first thing to check.
    """

    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 0.0):
        super().__init__()
        self.cell = LSTMCellScratch(input_size, hidden_size, forget_bias)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state: Optional[State] = None
                ) -> Tuple[torch.Tensor, State]:
        T, B, _ = x.shape
        if state is None:
            zeros = x.new_zeros(B, self.hidden_size)
            state = (zeros, zeros)

        outputs: List[torch.Tensor] = []
        for t in range(T):
            state = self.cell(x[t], state)
            outputs.append(state[0])          # collect h_t

        # torch.stack builds the (T, B, H) output tensor. Every element of
        # `outputs` is still attached to the autograd graph, so backprop
        # through this stack is backprop-through-time.
        return torch.stack(outputs, dim=0), state


class DeepLSTMScratch(nn.Module):
    """A stack of L LSTM layers — the paper's "deep LSTM".

    Layer 0 reads the embeddings. Layer k>0 reads the *output sequence* of
    layer k-1. Each layer keeps its own (h, c).

    WHY DEPTH HELPS (paper §3.4: "we found that deep LSTMs significantly
    outperformed shallow LSTMs, where each additional layer reduced perplexity
    by nearly 10%"): one LSTM layer can only apply one elementwise-gated affine
    step per timestep. Stacking gives the network a *vertical* computation
    budget per token that is independent of sequence length — layer 0 can do
    something lexical, layer 3 something syntactic.

    The returned `states` list is the thing the whole paper hinges on: after
    the encoder consumes the source sentence, `states` IS the fixed-size
    vector v of Eq. (1). For 4 layers x 1000 units x (h and c) that is
    4*1000*2 = 8000 numbers, and the entire source sentence must fit in it.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.0, forget_bias: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            LSTMLayerScratch(input_size if k == 0 else hidden_size,
                             hidden_size, forget_bias)
            for k in range(num_layers)
        ])
        # Dropout is applied BETWEEN layers only, never along the time axis —
        # dropping a different subset of units at every timestep would inject
        # noise into the memory highway itself and destroy long-range recall
        # (Zaremba et al. 2014). The 2014 seq2seq paper uses no dropout; we
        # default it off and expose it for the small-data setting we train in.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, states: Optional[List[State]] = None
                ) -> Tuple[torch.Tensor, List[State]]:
        if states is None:
            states = [None] * self.num_layers  # type: ignore[list-item]

        new_states: List[State] = []
        out = x
        for k, layer in enumerate(self.layers):
            if self.dropout is not None and k > 0:
                out = self.dropout(out)
            out, st = layer(out, states[k])
            new_states.append(st)
        return out, new_states


# --------------------------------------------------------------------------
# Proof of correctness: are we the same function as nn.LSTM?
# --------------------------------------------------------------------------

def copy_weights_from_torch(scratch: DeepLSTMScratch, ref: nn.LSTM) -> None:
    """Copy every parameter of an `nn.LSTM` into our stack.

    This works with zero reshaping *only because* we chose the same fat-matrix
    layout and the same i,f,g,o chunk order. If you ever see a scratch LSTM
    that "almost" matches a reference, a permuted gate order is the first
    suspect.
    """
    for k, layer in enumerate(scratch.layers):
        with torch.no_grad():
            layer.cell.weight_ih.copy_(getattr(ref, f"weight_ih_l{k}"))
            layer.cell.weight_hh.copy_(getattr(ref, f"weight_hh_l{k}"))
            layer.cell.bias_ih.copy_(getattr(ref, f"bias_ih_l{k}"))
            layer.cell.bias_hh.copy_(getattr(ref, f"bias_hh_l{k}"))


def verify_against_torch(T: int = 17, B: int = 5, I: int = 11, H: int = 13,
                         L: int = 4, device: str = "cpu",
                         dtype: torch.dtype = torch.float64) -> dict:
    """Run both implementations on identical inputs and weights; compare.

    We compare BOTH directions:
      * forward  — do we produce the same outputs and the same final (h, c)?
      * backward — do we produce the same *gradients*? A forward-only match can
                   hide a bug that autograd would otherwise expose (e.g. an
                   accidental .detach() that silently cuts backprop-through-
                   time). Checking gradients is the real test.

    float64 is deliberate: in float32 the two implementations differ by ~1e-6
    purely from GEMM reduction order, which makes "is it correct?" a judgement
    call. In float64 a correct implementation agrees to ~1e-13 and a wrong one
    does not, so the test has a clean answer.
    """
    torch.manual_seed(0)
    ref = nn.LSTM(I, H, num_layers=L).to(device=device, dtype=dtype)
    mine = DeepLSTMScratch(I, H, L).to(device=device, dtype=dtype)
    copy_weights_from_torch(mine, ref)

    x = torch.randn(T, B, I, device=device, dtype=dtype)
    x_ref, x_mine = x.clone().requires_grad_(), x.clone().requires_grad_()

    y_ref, (h_ref, c_ref) = ref(x_ref)
    y_mine, st_mine = mine(x_mine)
    h_mine = torch.stack([s[0] for s in st_mine])
    c_mine = torch.stack([s[1] for s in st_mine])

    # A scalar that depends on every output element, so every path gets a
    # gradient. (A plain .sum() would too, but a random projection also catches
    # sign errors that a symmetric sum could cancel out.)
    torch.manual_seed(1)
    w = torch.randn_like(y_ref)
    (y_ref * w).sum().backward()
    (y_mine * w).sum().backward()

    def gap(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item()

    grad_gap = max(
        gap(getattr(ref, f"weight_ih_l{k}").grad, mine.layers[k].cell.weight_ih.grad)
        for k in range(L)
    )
    return {
        "output_max_abs_diff": gap(y_ref, y_mine),
        "h_final_max_abs_diff": gap(h_ref, h_mine),
        "c_final_max_abs_diff": gap(c_ref, c_mine),
        "input_grad_max_abs_diff": gap(x_ref.grad, x_mine.grad),
        "weight_grad_max_abs_diff": grad_gap,
    }


# --------------------------------------------------------------------------
# The vanishing-gradient demo: why not a plain RNN?
# --------------------------------------------------------------------------

class VanillaRNNLayer(nn.Module):
    """h_t = tanh(W_hh h_{t-1} + W_ih x_t + b). The thing LSTMs replaced."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        std = 1.0 / math.sqrt(hidden_size)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        T, B, _ = x.shape
        if h is None:
            h = x.new_zeros(B, self.hidden_size)
        outs = []
        for t in range(T):
            h = torch.tanh(x[t] @ self.weight_ih.t() + h @ self.weight_hh.t() + self.bias)
            outs.append(h)
        return torch.stack(outs), h


def grad_flow_demo(T: int = 100, B: int = 4, H: int = 64, device: str = "cpu",
                   forget_biases: Optional[List[float]] = None) -> dict:
    """Measure how much gradient survives the trip from step T back to step t.

    METHOD. Feed a length-T sequence of inputs that require grad. Put a loss on
    the LAST timestep only. Then ||d loss / d x_t|| says how much influence the
    input at time t still has on the final output — i.e. how far back the model
    could possibly learn. We normalise so the last step reads 1.0.

    WHAT YOU WILL ACTUALLY SEE (and the honest version of the textbook story).
    At *initialisation* both curves decay. The LSTM decays more slowly — about
    three orders of magnitude better by 30 steps — but it does decay, and a
    beginner who was promised "LSTMs solve vanishing gradients" is entitled to
    ask why.

    The answer is that a freshly initialised LSTM has all its pre-activations
    near zero, so the forget gate sits at f = sigmoid(0) = 0.5. The memory line
    therefore multiplies by 0.5 every step: 0.5^30 ~ 1e-9. The decay is not a
    property of the architecture, it is a property of *this particular point*
    in its parameter space.

    The architecture's real gift is that a NON-decaying point EXISTS and is
    reachable. Push the forget-gate bias up and f moves toward 1: at bias 3,
    f = sigmoid(3) = 0.95 and 0.95^30 = 0.21 — the gradient arrives basically
    intact. Gradient descent can find that region, because f is produced by
    learnable parameters. A vanilla RNN has no such region: making the
    recurrence non-decaying means pushing W's spectral radius to 1, which makes
    the forward pass blow up in every direction it is not needed.

    So we sweep the forget-gate bias, and let the reader watch a knob turn the
    vanishing gradient off.

    THIS IS THE WHOLE PAPER IN ONE PLOT. Sutskever et al. need the encoder's
    first word to still influence the decoder's first word, which for a 25-word
    sentence is a 25+ step gap. Reversing the source (§3.3) shrinks that gap for
    the *early* words to ~0, which is exactly why it helps so much: it moves the
    hardest dependencies into the part of the curve that has not decayed yet.
    """
    if forget_biases is None:
        forget_biases = [0.0, 1.0, 3.0]
    torch.manual_seed(0)
    x = torch.randn(T, B, H, device=device)

    def run(model) -> List[float]:
        inp = x.clone().requires_grad_()
        out, _ = model(inp)
        out[-1].pow(2).sum().backward()
        g = inp.grad
        per_t = [g[t].norm().item() for t in range(T)]
        last = per_t[-1] if per_t[-1] > 0 else 1.0
        return [v / last for v in per_t]

    torch.manual_seed(0)
    out = {"T": T, "rnn": run(VanillaRNNLayer(H, H).to(device)), "lstm": {}}
    for fb in forget_biases:
        torch.manual_seed(0)   # identical weights; ONLY the forget bias differs
        out["lstm"][str(fb)] = run(LSTMLayerScratch(H, H, forget_bias=fb).to(device))
    return out
