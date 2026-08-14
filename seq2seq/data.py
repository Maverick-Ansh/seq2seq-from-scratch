"""
seq2seq.data — turning sentences into integer tensors, the paper's way.

The paper trains on WMT'14 English→French: 12M sentence pairs, 348M French
words, source vocabulary 160,000 and target vocabulary 80,000. We are on two
T4s for one session, so we use **Multi30k** — 29,000 English/French sentence
pairs of image captions. Same *shape* of problem, 400x less of it.

Everything in this file that matters is a decision the paper made explicitly:

  * a closed vocabulary of the N most frequent words, everything else -> UNK
    (§3.1: "we used the 160,000 most frequent words for the source language
     and the 80,000 most frequent words for the target language. Every
     out-of-vocabulary word was replaced with a special 'UNK' token.")
  * an <eos> symbol on every sentence, which is what "lets the model define a
    distribution over sequences of all possible lengths" (§2)
  * optionally REVERSING the source word order (§3.3) — the paper's headline
    trick, which we will A/B in Chapter 3
  * minibatches whose sentences all have roughly the same length (§3.4:
    "yielding a 2x speedup") — we measure that claim in Chapter 5
"""

from __future__ import annotations

import collections
import gzip
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import torch

MULTI30K_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

# Which files make up which split. Multi30k ships several test sets; the 2016
# Flickr one is the standard "test" everyone reports on.
SPLIT_FILES = {
    "train": "train",
    "valid": "val",
    "test": "test_2016_flickr",
}

# ---------------------------------------------------------------------------
# Special symbols
# ---------------------------------------------------------------------------
# The integer values are fixed and meaningful:
#   PAD = 0  so that a zero-filled tensor is already "all padding", and so that
#            `ignore_index=0` in the loss is the natural thing to write.
#   SOS      the token we feed the decoder at its first step. (The paper is
#            slightly different: it feeds the encoder's <eos> as the decoder's
#            first input. Using a distinct <sos> is the modern convention and
#            is strictly more expressive — the decoder can then tell "I am
#            starting" apart from "the source just ended". We note the
#            deviation here rather than hide it.)
#   EOS      appended to every sentence, source and target. On the target side
#            it is what the model must learn to emit to STOP; without it,
#            generation has no defined end.
#   UNK      every word outside the closed vocabulary.
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Deliberately a 3-line regex rather than spaCy/Moses. Two reasons:
#   1. You can read it. A tokeniser you cannot read is a place bugs hide.
#   2. It has no dependencies, so this repo runs on an air-gapped machine.
# It lowercases (halving vocabulary size, which matters a lot at 29k sentences)
# and splits punctuation off as its own token, so "chien." -> ["chien", "."].
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower().strip())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """A closed word<->id mapping, built from training-set frequencies only.

    WHY "TRAINING SET ONLY" IS NOT A DETAIL. If you build the vocabulary from
    train+valid+test, then a word that appears only in the test set gets a real
    embedding instead of <unk>, and your test score is measuring a model that
    peeked. That is one of the most common silent evaluation leaks in NLP.
    """

    def __init__(self, counter: collections.Counter, max_size: int, min_freq: int = 1):
        self.itos: List[str] = list(SPECIALS)
        # most_common() sorts by count; ties are broken by insertion order,
        # which is the order words were first seen -> deterministic across runs.
        for word, freq in counter.most_common():
            if len(self.itos) >= max_size:
                break
            if freq < min_freq:
                break
            self.itos.append(word)
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.counter = counter

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        s = self.stoi
        return [s.get(t, UNK) for t in tokens]

    def decode(self, ids: Sequence[int], strip_specials: bool = True) -> List[str]:
        out = []
        for i in ids:
            if strip_specials and i in (PAD, SOS, EOS):
                continue
            out.append(self.itos[i])
        return out

    def coverage(self, counter: collections.Counter) -> float:
        """Fraction of running-text tokens this vocab can represent (not <unk>).

        Report this, always. A vocabulary that covers 95% of tokens leaves one
        <unk> in every 20 words — i.e. in most sentences. The paper's 160k/80k
        vocabularies exist precisely to push this number up, and its BLEU is
        held back by the <unk>s that remain (§3.6 discusses exactly this).
        """
        total = sum(counter.values())
        known = sum(c for w, c in counter.items() if w in self.stoi)
        return known / max(total, 1)


# ---------------------------------------------------------------------------
# Loading Multi30k
# ---------------------------------------------------------------------------

def download_multi30k(root: str, langs: Tuple[str, str] = ("en", "fr")) -> None:
    os.makedirs(root, exist_ok=True)
    for split, stem in SPLIT_FILES.items():
        for lang in langs:
            name = f"{stem}.{lang}"
            dest = os.path.join(root, name)
            if os.path.exists(dest):
                continue
            url = f"{MULTI30K_BASE}/{name}.gz"
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            with open(dest, "wb") as f:
                f.write(gzip.decompress(blob))


def read_split(root: str, split: str, langs: Tuple[str, str]) -> List[Tuple[List[str], List[str]]]:
    stem = SPLIT_FILES[split]
    with open(os.path.join(root, f"{stem}.{langs[0]}"), encoding="utf-8") as f:
        src = f.read().splitlines()
    with open(os.path.join(root, f"{stem}.{langs[1]}"), encoding="utf-8") as f:
        tgt = f.read().splitlines()
    assert len(src) == len(tgt), f"{split}: {len(src)} source lines vs {len(tgt)} target"
    pairs = [(tokenize(s), tokenize(t)) for s, t in zip(src, tgt)]
    # Drop empty lines — a zero-length source has no encoder input at all and
    # would produce a NaN the first time you hit it.
    return [(s, t) for s, t in pairs if s and t]


@dataclass
class Corpus:
    train: List[Tuple[List[str], List[str]]]
    valid: List[Tuple[List[str], List[str]]]
    test: List[Tuple[List[str], List[str]]]
    src_vocab: Vocab
    tgt_vocab: Vocab
    langs: Tuple[str, str]

    def stats(self) -> dict:
        def tok_count(pairs, i):
            return sum(len(p[i]) for p in pairs)
        src_test = collections.Counter(w for s, _ in self.test for w in s)
        tgt_test = collections.Counter(w for _, t in self.test for w in t)
        return {
            "langs": f"{self.langs[0]}->{self.langs[1]}",
            "sentences": {"train": len(self.train), "valid": len(self.valid),
                          "test": len(self.test)},
            "train_tokens": {self.langs[0]: tok_count(self.train, 0),
                             self.langs[1]: tok_count(self.train, 1)},
            "vocab": {self.langs[0]: len(self.src_vocab), self.langs[1]: len(self.tgt_vocab)},
            "test_token_coverage": {self.langs[0]: round(self.src_vocab.coverage(src_test), 4),
                                    self.langs[1]: round(self.tgt_vocab.coverage(tgt_test), 4)},
            "mean_len": {self.langs[0]: round(tok_count(self.train, 0) / len(self.train), 2),
                         self.langs[1]: round(tok_count(self.train, 1) / len(self.train), 2)},
        }


def load_corpus(root: str = "/kaggle/working/data/multi30k",
                langs: Tuple[str, str] = ("en", "fr"),
                src_vocab_size: int = 10_000,
                tgt_vocab_size: int = 10_000,
                min_freq: int = 2) -> Corpus:
    download_multi30k(root, langs)
    train = read_split(root, "train", langs)
    valid = read_split(root, "valid", langs)
    test = read_split(root, "test", langs)

    src_counter = collections.Counter(w for s, _ in train for w in s)
    tgt_counter = collections.Counter(w for _, t in train for w in t)
    return Corpus(
        train=train, valid=valid, test=test, langs=langs,
        src_vocab=Vocab(src_counter, src_vocab_size, min_freq),
        tgt_vocab=Vocab(tgt_counter, tgt_vocab_size, min_freq),
    )


# ---------------------------------------------------------------------------
# Numericalisation: sentences -> id lists
# ---------------------------------------------------------------------------

def encode_pair(src_tokens: Sequence[str], tgt_tokens: Sequence[str],
                corpus: Corpus, reverse_source: bool) -> Tuple[List[int], List[int]]:
    """One (source, target) pair as integer ids.

    Source  : [w_T ... w_1] + [EOS]     if reverse_source else  [w_1 ... w_T] + [EOS]
    Target  : [SOS] + [y_1 ... y_T'] + [EOS]

    THE REVERSAL (paper §3.3). We reverse the *words*, then append <eos>; the
    end-of-input marker stays at the end where the encoder can use it as "stop
    reading". The paper reverses the source and NOT the target — that asymmetry
    is the entire point, and reversing both would undo the benefit exactly.

    Why it helps (worth internalising before Chapter 3): with the normal order,
    x_1 is T steps away from y_1, x_2 is T steps from y_2, ... every source word
    is equidistant-far from its translation. Average distance T, minimum
    distance T. Reverse the source and x_1 sits *adjacent* to y_1, x_2 next to
    y_2, and so on for the early words. The average distance is unchanged — the
    late words got worse by exactly as much as the early words got better — but
    the MINIMUM distance collapses from T to ~0. The paper calls this reducing
    the "minimal time lag", and argues optimisation cares about the minimum,
    not the mean: once backprop establishes *any* short path between source and
    target, the rest follows. Chapter 1's gradient-flow plot is why.

    The target keeps its natural order because the decoder must produce
    left-to-right French, and because at inference time we have no target to
    reverse.
    """
    src = list(src_tokens)[::-1] if reverse_source else list(src_tokens)
    src_ids = corpus.src_vocab.encode(src) + [EOS]
    tgt_ids = [SOS] + corpus.tgt_vocab.encode(tgt_tokens) + [EOS]
    return src_ids, tgt_ids


def encode_split(pairs, corpus: Corpus, reverse_source: bool
                 ) -> List[Tuple[List[int], List[int]]]:
    return [encode_pair(s, t, corpus, reverse_source) for s, t in pairs]


# ---------------------------------------------------------------------------
# Batching: the paper's "sentences of roughly the same length"
# ---------------------------------------------------------------------------

def pad_batch(examples: Sequence[Tuple[List[int], List[int]]], device="cpu"
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of variable-length pairs into rectangular tensors.

    Returns (src, tgt_in, tgt_out), all TIME-FIRST:
        src     (S, B)  encoder input
        tgt_in  (T, B)  decoder input  = [SOS] y_1 ... y_{T'-1}
        tgt_out (T, B)  decoder target =  y_1 ... y_{T'}  EOS

    THE OFF-BY-ONE THAT DEFINES TEACHER FORCING. tgt_in and tgt_out are the
    same sequence shifted by one position. At step t the decoder is *shown*
    tgt_in[t] (the true previous word) and must *predict* tgt_out[t] (the true
    next word). Feeding the ground-truth previous word rather than the model's
    own previous guess is called TEACHER FORCING: it makes all T steps of the
    loss computable in one parallel pass, and it stops early mistakes from
    derailing the whole sentence during training.

    Its cost is "exposure bias": at inference the model must consume its own
    outputs, a distribution it never saw in training. Beam search (Chapter 6)
    is partly a patch for exactly this.

    On the target side, PAD=0 and the loss is told to ignore index 0, so
    padded positions contribute nothing to the gradient.

    The SOURCE side has no such escape. There is no attention in this model —
    the encoder's final state *is* the entire message — so any padding the
    encoder consumes gets baked into that state. Left-padding is what saves us:
    the PADs are consumed FIRST, while the state is still near zero, and the
    real sentence is the last thing the encoder sees. Combine that with
    bucketing (below) so batches are near-uniform in length, and the amount of
    padding the encoder ever eats is small.
    """
    S = max(len(s) for s, _ in examples)
    T = max(len(t) for _, t in examples)
    B = len(examples)
    src = torch.full((S, B), PAD, dtype=torch.long)
    tgt = torch.full((T, B), PAD, dtype=torch.long)
    for b, (s, t) in enumerate(examples):
        # LEFT-pad the source, right-pad the target. Left-padding the source
        # means every sentence's real final token — and therefore the state the
        # decoder inherits — lands on the SAME timestep S-1 for the whole
        # batch. Right-padding would leave shorter sentences' useful state
        # buried mid-sequence and then smeared by PAD embeddings.
        src[S - len(s):, b] = torch.tensor(s)
        tgt[:len(t), b] = torch.tensor(t)
    return (src.to(device), tgt[:-1].to(device), tgt[1:].to(device))


def bucket_batches(examples: Sequence[Tuple[List[int], List[int]]],
                   batch_size: int, shuffle: bool = True,
                   seed: int = 0, bucket_multiplier: int = 50
                   ) -> List[List[Tuple[List[int], List[int]]]]:
    """Group sentences of similar length into the same minibatch (paper §3.4).

    THE PROBLEM. A minibatch is a rectangle. If one sentence has 40 tokens and
    the rest have 6, the rectangle is 40 wide and ~85% of the GPU's work is
    spent multiplying padding by weights. Worse for an RNN than for a
    feed-forward net, because the number of *sequential timesteps* is set by the
    longest member: the batch does not just waste FLOPs, it waits.

    THE FIX, and why it is not just "sort everything". Perfect sorting gives
    minimum padding but destroys shuffling: every epoch sees the identical
    batches in the identical order, and each batch is topically homogeneous
    (all the 5-word sentences together), which biases the gradient. The
    standard compromise, used here:

        1. shuffle the whole dataset
        2. cut it into "pools" of  batch_size * bucket_multiplier  examples
        3. sort WITHIN a pool by length
        4. cut each pool into batches -> each batch is nearly uniform in length
        5. shuffle the ORDER of the resulting batches

    You get ~all of the padding savings with ~all of the shuffling randomness.
    `bucket_multiplier` is the knob: 1 = pure shuffle (max padding), huge =
    global sort (min padding, least randomness).

    We sort by (source length, target length): the source length sets the
    encoder rectangle and is the dominant cost, target length breaks ties.
    """
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)

    pool_size = max(batch_size * bucket_multiplier, batch_size)
    batches: List[List[Tuple[List[int], List[int]]]] = []
    for start in range(0, len(idx), pool_size):
        pool = idx[start:start + pool_size]
        pool.sort(key=lambda i: (len(examples[i][0]), len(examples[i][1])))
        for b in range(0, len(pool), batch_size):
            chunk = pool[b:b + batch_size]
            if chunk:
                batches.append([examples[i] for i in chunk])
    if shuffle:
        rng.shuffle(batches)
    return batches


def padding_stats(batches) -> dict:
    """How much of the tensor we build is real data vs padding.

    `token_efficiency` is the number to watch: real tokens / rectangle area.
    `timesteps` is the other one — for an RNN, total sequential steps executed
    is sum(max_len) over batches, and that is what wall-clock actually tracks.
    """
    real = pad = 0
    steps = 0
    for batch in batches:
        S = max(len(s) for s, _ in batch)
        T = max(len(t) for _, t in batch)
        B = len(batch)
        real += sum(len(s) for s, _ in batch) + sum(len(t) for _, t in batch)
        pad += S * B + T * B
        steps += S + T
    return {"real_tokens": real, "rectangle_cells": pad,
            "token_efficiency": round(real / pad, 4),
            "sequential_timesteps": steps, "num_batches": len(batches)}


def iterate(examples, batch_size: int, device, shuffle: bool = True,
            seed: int = 0, bucket_multiplier: int = 50) -> Iterator:
    for batch in bucket_batches(examples, batch_size, shuffle, seed, bucket_multiplier):
        yield pad_batch(batch, device)
