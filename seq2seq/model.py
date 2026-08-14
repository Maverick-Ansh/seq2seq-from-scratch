"""
seq2seq.model — the paper's architecture, Equation 1 made executable.

    p(y_1..y_T' | x_1..x_T)  =  PROD_t  p(y_t | v, y_1..y_{t-1})

where v is the encoder LSTM's final state. Three components:

    ENCODER   a deep LSTM that reads the source and is then thrown away except
              for its final (h, c) — that pair IS v.
    DECODER   a SECOND deep LSTM, initialised with v, that is a conditional
              language model over the target.
    GENERATOR a linear map from decoder hidden state to vocabulary logits,
              followed by softmax.

DESIGN CHOICES, EACH TRACEABLE TO A SENTENCE IN THE PAPER
---------------------------------------------------------
* TWO SEPARATE LSTMs, not one shared. §2: "we used two different LSTMs: one for
  the input sequence and another for the output sequence, because doing so
  increases the number of model parameters at negligible computational cost and
  makes it natural to train the LSTM on multiple language pairs
  simultaneously." Two 32M-parameter recurrent stacks instead of one; the
  encoder is free to specialise in *compressing* English and the decoder in
  *generating* French, which are not the same job.

* FOUR LAYERS. §3.4: "deep LSTMs significantly outperformed shallow LSTMs,
  where each additional layer reduced perplexity by nearly 10%".

* INITIALISATION U(-0.08, 0.08) on every parameter (§3.4). Note this is
  NOT the usual 1/sqrt(fan_in) scaling: for H=1000, 1/sqrt(1000) = 0.032, so
  0.08 is about 2.5x larger than the standard choice. A larger init means
  larger initial gate pre-activations, which pushes the forget gate away from
  the f = 0.5 dead zone we measured in Chapter 1. Combined with gradient
  clipping (which makes a too-large init survivable), this is a coherent
  package, and it is why you should not mix-and-match hyperparameters across
  papers.

* NO ATTENTION. Attention did not exist yet (Bahdanau et al. appeared months
  later). The consequence is severe and worth staring at: the ONLY channel
  from source to target is v, a fixed-size vector, regardless of whether the
  source is 3 words or 80. Everything the decoder will ever know about the
  English sentence must fit in 4 layers x 1000 units x 2 tensors = 8000
  numbers. This is "the bottleneck", and removing it is what the next three
  years of the field were about.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EOS, PAD, SOS
from .lstm import DeepLSTMScratch

State = Tuple[torch.Tensor, torch.Tensor]


class Seq2Seq(nn.Module):
    """Encoder LSTM -> fixed vector v -> decoder LSTM -> softmax.

    Args:
        src_vocab, tgt_vocab : vocabulary sizes
        emb_dim, hidden_dim  : the paper uses 1000 for both
        num_layers           : the paper uses 4
        scratch              : if True, use our hand-written Python-loop LSTM
                               from Chapter 1. If False, use cuDNN's nn.LSTM.
                               We PROVED in Chapter 1 that these compute the
                               same function, so switching to the fast one is
                               not a compromise — it is the payoff for having
                               done the proof. (The scratch loop is ~10-30x
                               slower because every timestep is a separate
                               Python-level CUDA launch; cuDNN fuses the whole
                               sequence into one kernel.)
    """

    def __init__(self, src_vocab: int, tgt_vocab: int, emb_dim: int = 512,
                 hidden_dim: int = 512, num_layers: int = 4, dropout: float = 0.0,
                 scratch: bool = False, init_range: float = 0.08):
        super().__init__()
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.scratch = scratch

        # padding_idx=PAD makes the PAD row a permanent zero vector whose
        # gradient is always zero — padding cannot drift into meaning anything.
        self.src_embed = nn.Embedding(src_vocab, emb_dim, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, emb_dim, padding_idx=PAD)

        if scratch:
            self.encoder = DeepLSTMScratch(emb_dim, hidden_dim, num_layers, dropout)
            self.decoder = DeepLSTMScratch(emb_dim, hidden_dim, num_layers, dropout)
        else:
            self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
            self.decoder = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)

        # The generator. In the paper this is a 1000 x 80000 matrix — 80M
        # parameters, more than the entire recurrent network, and the single
        # most expensive op in the model. §3.5 shards exactly this across 4
        # GPUs. We do the same in Chapter 4.
        self.generator = nn.Linear(hidden_dim, tgt_vocab)

        self.apply_paper_init(init_range)

    def apply_paper_init(self, r: float = 0.08) -> None:
        """§3.4: 'we initialized all of the LSTM's parameters with the uniform
        distribution between -0.08 and 0.08'. All of them — embeddings,
        recurrent weights, biases, output layer."""
        for p in self.parameters():
            nn.init.uniform_(p, -r, r)
        with torch.no_grad():                      # restore the PAD zero rows
            self.src_embed.weight[PAD].zero_()
            self.tgt_embed.weight[PAD].zero_()

    # -- the two halves -----------------------------------------------------

    def encode(self, src: torch.Tensor):
        """src (S, B) -> v, the encoder's final state.

        Note what is NOT returned: the per-timestep outputs. In this
        architecture they are computed and immediately discarded. An
        attention model would keep every one of them; that difference is
        the whole of Bahdanau 2015 in one line.
        """
        emb = self.src_embed(src)                  # (S, B, E)
        _, v = self.encoder(emb)                   # v = (h, c), each (L, B, H)
        return v

    def decode(self, tgt_in: torch.Tensor, v):
        """tgt_in (T, B) + initial state v -> logits (T, B, V).

        TEACHER FORCING lives here. tgt_in is the *ground truth* shifted right,
        so all T steps can be run in a single call: step t is shown the true
        y_{t-1}, never its own guess. That makes the whole target sequence one
        parallel cuDNN call instead of T dependent ones.
        """
        emb = self.tgt_embed(tgt_in)               # (T, B, E)
        out, _ = self.decoder(emb, v)              # (T, B, H)
        return self.generator(out)                 # (T, B, V)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(tgt_in, self.encode(src))

    # -- one step at a time, for beam search --------------------------------

    def decode_step(self, y_prev: torch.Tensor, state):
        """One decoder timestep. y_prev (B,) -> log-probs (B, V), new state.

        Used at INFERENCE, where teacher forcing is impossible: there is no
        ground-truth previous word, only the one we just chose. This is the
        train/test mismatch called *exposure bias*.
        """
        emb = self.tgt_embed(y_prev).unsqueeze(0)  # (1, B, E)
        out, state = self.decoder(emb, state)
        logits = self.generator(out.squeeze(0))    # (B, V)
        return F.log_softmax(logits, dim=-1), state


class EnsembleSeq2Seq(nn.Module):
    """Several independently-trained Seq2Seq models decoded as one.

    §3.6 reports the ensemble as the difference between beating the SMT
    baseline and not: a single reversed LSTM scores 30.59 BLEU (below the
    33.30 baseline), while an ensemble of 5 scores 34.81.

    HOW ENSEMBLING WORKS HERE. At each decoding step, every member produces a
    distribution over the vocabulary. We average them and hand the result to
    the beam. Averaging in LOG space (a geometric mean of probabilities, then
    renormalised) rather than probability space is the standard choice, and the
    difference matters: the geometric mean is small whenever ANY member says
    "unlikely", so a single confident objection can veto a word. An arithmetic
    mean lets one enthusiastic member carry a word through. For translation the
    conservative option is the better one.

    WHY IT HELPS. Each model makes different mistakes — different random init,
    different batch order. Errors that are uncorrelated across members partly
    cancel when averaged, while the signal, which is shared, does not. The
    catch is in the word "uncorrelated": ensembling N copies of the same
    training run with the same seed buys nothing. Diversity is the whole asset.

    IMPLEMENTATION NOTE. We concatenate the members' states along the LAYER
    axis, so an ensemble's state has the same rank and shape convention as a
    single model's (L*M, B, H). That lets `beam.beam_search` reorder and
    repeat_interleave the state with no special-casing — the ensemble is a
    drop-in for a single model.
    """

    def __init__(self, models):
        super().__init__()
        assert len(models) > 0
        self.models = nn.ModuleList(models)
        self.tgt_vocab = models[0].tgt_vocab
        self.num_layers = models[0].num_layers

    def encode(self, src):
        hs, cs = zip(*[m.encode(src) for m in self.models])
        return torch.cat(hs, dim=0), torch.cat(cs, dim=0)

    def decode_step(self, y_prev, state):
        h, c = state
        L = self.num_layers
        logps, new_h, new_c = [], [], []
        for i, m in enumerate(self.models):
            sl = slice(i * L, (i + 1) * L)
            lp, (hi, ci) = m.decode_step(y_prev, (h[sl].contiguous(), c[sl].contiguous()))
            logps.append(lp)
            new_h.append(hi)
            new_c.append(ci)
        # Mean of log-probs, then renormalise so the result is a proper
        # distribution again (the mean of log-probs is not normalised).
        mean_lp = torch.stack(logps).mean(0)
        mean_lp = mean_lp - mean_lp.logsumexp(dim=-1, keepdim=True)
        return mean_lp, (torch.cat(new_h, dim=0), torch.cat(new_c, dim=0))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sequence_loss(logits: torch.Tensor, target: torch.Tensor,
                  label_smoothing: float = 0.0) -> torch.Tensor:
    """Mean cross-entropy per non-pad token.

    The paper's objective (§3.2) is

        (1/|S|) * SUM_{(T,S) in S} log p(T|S)

    i.e. the mean log-probability of the correct translation. Since
    log p(T|S) = SUM_t log p(y_t | ...), maximising it is exactly minimising
    token-level cross-entropy — the loss below. The paper normalises per
    SENTENCE; we normalise per TOKEN, which is the standard modern choice
    because it stops long sentences from dominating the gradient, and because
    exp(per-token loss) is *perplexity*, the quantity §3.3 reports (5.8 -> 4.7
    from reversing the source).

    ignore_index=PAD is not optional. Without it the model is rewarded for
    confidently predicting padding, which it can do perfectly, which drags the
    loss down and makes your metric a lie.
    """
    V = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, V), target.reshape(-1),
                           ignore_index=PAD, label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------

def count_parameters(model: Seq2Seq) -> dict:
    """Break the parameter count down by role.

    Do this for every model you build. It tells you instantly where your memory
    and compute actually go, and it is usually somewhere you did not expect —
    in this architecture the *embeddings and the output softmax* dwarf the
    recurrent network that everyone talks about.
    """
    def n(mod):
        return sum(p.numel() for p in mod.parameters())
    parts = {
        "src_embedding": n(model.src_embed),
        "tgt_embedding": n(model.tgt_embed),
        "encoder_lstm": n(model.encoder),
        "decoder_lstm": n(model.decoder),
        "generator_softmax": n(model.generator),
    }
    parts["total"] = sum(parts.values())
    parts["recurrent_only"] = parts["encoder_lstm"] + parts["decoder_lstm"]
    return parts


def paper_config() -> dict:
    """The exact configuration of §3.4, for the memory experiment in Ch. 4."""
    return dict(src_vocab=160_000, tgt_vocab=80_000, emb_dim=1000,
                hidden_dim=1000, num_layers=4)
