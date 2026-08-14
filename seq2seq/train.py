"""
seq2seq.train — the paper's training recipe, §3.4, with every knob explained.

THE RECIPE, quoted:

    "We initialized all of the LSTM's parameters with the uniform distribution
     between -0.08 and 0.08. We used stochastic gradient descent without
     momentum, with a fixed learning rate of 0.7. After 5 epochs, we begun
     halving the learning rate every half epoch. We trained our models for a
     total of 7.5 epochs. We used batches of 128 sequences for the gradient and
     divided it by the size of the batch (namely, 128). Although LSTMs tend not
     to suffer from the vanishing gradient problem, they can have exploding
     gradients. Thus we enforced a hard constraint on the norm of the gradient
     by scaling it when its norm exceeded a threshold. For each training batch,
     we compute s = ||g||_2, where g is the gradient divided by 128. If s > 5,
     we set g = 5g/s."

Four things in there, each of which is a real decision:

1. SGD WITHOUT MOMENTUM at a FIXED lr of 0.7. In 2026 you would reach for Adam
   at 3e-4 without thinking. 0.7 looks insane next to that — but it is not
   comparable, because Adam divides by a running estimate of the gradient
   magnitude and plain SGD does not. SGD's step size must absorb the raw scale
   of the gradient. Combined with the clip below, a large lr with a hard
   ceiling on step size is a coherent package. Do not lift one number out of it.

2. A STEP SCHEDULE: constant for 5 epochs, then halve every half epoch for 2.5
   more. The first phase makes progress; the second anneals. This is the
   ancestor of every cosine/linear-decay schedule you have ever used, and the
   reason all of them exist is the same: a step size that finds the basin is
   too big to sit at the bottom of it.

3. GRADIENT CLIPPING BY GLOBAL NORM. Note what it is NOT: it is not clipping
   each coordinate. Scaling the whole gradient vector by 5/s preserves its
   DIRECTION and only shortens it. Per-coordinate clipping would change the
   direction and is a different (worse) algorithm.
   Note also the paper's own framing — "LSTMs tend not to suffer from the
   vanishing gradient problem, they can have exploding gradients." Chapter 1
   showed why: the c-line's dc_t/dc_{t-1} = f_t is a product of numbers in
   [0,1] for the memory path, but the h-path still carries weight matrices, and
   a single bad batch can spike the norm by orders of magnitude. Clipping makes
   the large learning rate survivable.

4. A NORMALISATION SUBTLETY THAT MATTERS. "g is the gradient divided by 128"
   means their loss is the SUM over the batch's 128 sentences divided by 128 —
   a per-SENTENCE mean. We use a per-TOKEN mean, because exp(per-token loss) is
   exactly the perplexity §3.3 reports (5.8 -> 4.7). The two differ by a factor
   of roughly the average sentence length (~15-25). So THE THRESHOLD 5 DOES NOT
   TRANSFER: our gradients are ~20x smaller, and clipping at 5 would essentially
   never fire. We keep the paper's *mechanism* and re-tune the *threshold*, and
   we log how often it fires so you can see it working rather than trust us.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from . import data as D
from .model import Seq2Seq, sequence_loss


@dataclass
class TrainConfig:
    # --- the paper's numbers ---
    lr: float = 0.7                 # §3.4
    momentum: float = 0.0           # §3.4: "without momentum"
    batch_size: int = 128           # §3.4
    clip: float = 5.0               # §3.4 (threshold retuned, see module docstring)
    epochs: float = 7.5             # §3.4
    constant_epochs: float = 5.0    # §3.4: halving begins after 5
    halve_every: float = 0.5        # §3.4: "every half epoch"
    # How the loss is normalised BEFORE .backward(). This one line decides
    # whether the paper's lr=0.7 and clip=5 mean anything at all.
    #   "sentence" — sum over all tokens, divided by the number of SENTENCES.
    #                This is the paper: "we used batches of 128 sequences for
    #                the gradient and divided it by the size of the batch
    #                (namely, 128)". Gradient norms come out ~20x larger.
    #   "token"    — the modern default: divide by the number of TOKENS.
    # We report per-token loss either way, so perplexity stays comparable.
    grad_normalization: str = "sentence"
    # --- our adaptations, all declared ---
    reverse_source: bool = True     # §3.3, the thing Chapter 3 A/B-tests
    bucket_multiplier: int = 50
    label_smoothing: float = 0.0
    seed: int = 0
    device: str = "cuda:0"
    log_every: int = 0              # 0 = only log at epoch boundaries


def lr_at(epoch_float: float, cfg: TrainConfig) -> float:
    """The paper's schedule as a pure function of how far through training we are.

    Constant at `lr` until `constant_epochs`, then halve every `halve_every`.
    Written as a function of a float epoch (not a step counter) so it is
    correct regardless of dataset size or batch size — the same schedule
    applied to 12M sentences and to 29k.
    """
    if epoch_float < cfg.constant_epochs:
        return cfg.lr
    halvings = int((epoch_float - cfg.constant_epochs) / cfg.halve_every) + 1
    return cfg.lr * (0.5 ** halvings)


@torch.no_grad()
def evaluate(model: Seq2Seq, examples, cfg: TrainConfig) -> Dict[str, float]:
    """Token-level cross-entropy and perplexity on a held-out split.

    PERPLEXITY is exp(mean per-token cross-entropy). Interpret it as "how many
    words is the model effectively choosing between at each step". A perplexity
    of 1 is a model that knows exactly what comes next; a perplexity equal to
    the vocabulary size is a model that has learned nothing. It is the metric
    §3.3 uses for the reversal result (5.8 -> 4.7).

    We accumulate SUM of loss * ntokens and divide once at the end, rather than
    averaging per-batch means. Averaging means-of-different-sized-batches
    silently over-weights the small ones, which for length-bucketed data means
    over-weighting short sentences — the easy ones.
    """
    model.eval()
    total_loss, total_tok = 0.0, 0
    for src, tin, tout in D.iterate(examples, cfg.batch_size, cfg.device,
                                    shuffle=False, bucket_multiplier=10**9):
        logits = model(src, tin)
        ntok = (tout != D.PAD).sum().item()
        total_loss += sequence_loss(logits, tout).item() * ntok
        total_tok += ntok
    model.train()
    ce = total_loss / max(total_tok, 1)
    return {"loss": ce, "ppl": math.exp(min(ce, 20)), "tokens": total_tok}


def train(model: Seq2Seq, corpus, cfg: TrainConfig, verbose: bool = True) -> Dict:
    """Run the paper's recipe. Returns a history dict, and mutates `model`."""
    torch.manual_seed(cfg.seed)
    model.to(cfg.device).train()

    train_ex = D.encode_split(corpus.train, corpus, cfg.reverse_source)
    valid_ex = D.encode_split(corpus.valid, corpus, cfg.reverse_source)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    hist = {"epoch": [], "train_loss": [], "valid_loss": [], "valid_ppl": [],
            "lr": [], "clip_frac": [], "sec": [], "words_per_sec": [],
            "grad_norm_median": []}
    n_batches = len(D.bucket_batches(train_ex, cfg.batch_size,
                                     bucket_multiplier=cfg.bucket_multiplier))
    total_epochs = cfg.epochs
    ep = 0.0
    t_start = time.time()

    while ep < total_epochs - 1e-9:
        ep_int = int(ep)
        t0 = time.time()
        run_loss, run_tok, clipped, seen, words = 0.0, 0, 0, 0, 0
        grad_norms: List[float] = []
        # A different seed per epoch, so the pooled bucketing reshuffles.
        for src, tin, tout in D.iterate(train_ex, cfg.batch_size, cfg.device,
                                        shuffle=True, seed=cfg.seed * 1000 + ep_int,
                                        bucket_multiplier=cfg.bucket_multiplier):
            # Update the learning rate CONTINUOUSLY, not once per epoch: the
            # paper halves every HALF epoch, which is mid-epoch by definition.
            cur_lr = lr_at(ep_int + seen / n_batches, cfg)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            # `loss` is always per-token, so exp(loss) is perplexity and the
            # numbers we print are comparable to §3.3 regardless of the
            # normalisation we backprop through.
            loss = sequence_loss(model(src, tin), tout, cfg.label_smoothing)
            ntok_t = (tout != D.PAD).sum()
            if cfg.grad_normalization == "sentence":
                # Recover the SUM over tokens, then divide by the batch's
                # sentence count — exactly the paper's "divided it by 128".
                loss_for_grad = loss * ntok_t / src.size(1)
            else:
                loss_for_grad = loss
            loss_for_grad.backward()

            # §3.4's hard constraint. clip_grad_norm_ returns the norm BEFORE
            # clipping, which is exactly the `s` in the paper's formula, so we
            # can count how often the constraint actually binds.
            s = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            grad_norms.append(s.item())
            clipped += int(s.item() > cfg.clip)
            opt.step()

            ntok = ntok_t.item()
            run_loss += loss.item() * ntok
            run_tok += ntok
            words += src.numel() + tout.numel()
            seen += 1
            if cfg.log_every and seen % cfg.log_every == 0 and verbose:
                print(f"    ep {ep_int} step {seen}/{n_batches} "
                      f"loss {run_loss/run_tok:.3f} lr {cur_lr:.4f}")

        dt = time.time() - t0
        va = evaluate(model, valid_ex, cfg)
        hist["epoch"].append(ep_int + 1)
        hist["train_loss"].append(run_loss / run_tok)
        hist["valid_loss"].append(va["loss"])
        hist["valid_ppl"].append(va["ppl"])
        hist["lr"].append(cur_lr)
        hist["clip_frac"].append(clipped / max(seen, 1))
        hist["sec"].append(dt)
        hist["words_per_sec"].append(int(words / dt))
        grad_norms.sort()
        hist["grad_norm_median"].append(grad_norms[len(grad_norms) // 2])
        if verbose:
            print(f"  epoch {ep_int+1:>2}/{int(total_epochs)}  "
                  f"train {run_loss/run_tok:.3f}  valid {va['loss']:.3f}  "
                  f"ppl {va['ppl']:6.2f}  lr {cur_lr:.4f}  "
                  f"|g| {hist['grad_norm_median'][-1]:6.2f}  "
                  f"clip {100*clipped/max(seen,1):4.1f}%  "
                  f"{int(words/dt):,} w/s  {dt:.1f}s")
        ep += 1.0

    hist["total_sec"] = time.time() - t_start
    hist["config"] = cfg.__dict__.copy()
    return hist
