"""
seq2seq.bleu — BLEU from scratch (Papineni et al. 2002), the metric §3.6 reports.

BLEU asks: "of the n-grams the model produced, what fraction appear in the
reference — and is the output the right length?" Two components.

1. MODIFIED n-GRAM PRECISION, for n = 1,2,3,4.

   Plain precision would be (matching n-grams) / (produced n-grams). That is
   trivially gamed: for the reference "the cat sat on the mat", the hypothesis
   "the the the the the the" scores 6/6 = 1.0 on unigrams.

   The fix is CLIPPING. Each n-gram can only be credited as many times as it
   appears in the reference. "the" appears twice in the reference, so at most
   2 of the 6 "the"s count: 2/6 = 0.33. This clipping is the entire content of
   the word "modified", and it is the one part of BLEU people re-derive wrong.

   Precisions are combined with a GEOMETRIC mean, not an arithmetic one. That
   is deliberate and harsh: if any single n-gram order scores 0, the geometric
   mean is 0. A translation with no correct 4-gram gets BLEU 0 no matter how
   many words it got individually right.

2. BREVITY PENALTY.

   Precision alone rewards saying less — output one word you are sure of and
   score 1.0. BLEU has no recall term (with multiple references, recall is
   ill-defined), so it patches the hole with an explicit length penalty:

       BP = 1                    if c > r
       BP = exp(1 - r/c)         if c <= r

   where c = total hypothesis length, r = total reference length. Note it is
   one-sided: being too LONG is punished by precision itself (extra n-grams
   that match nothing), so BP only needs to punish being too SHORT.

   Remember this when you look at the beam-width sweep in Chapter 7: raw
   log-probability prefers short sentences, wider beams find shorter sentences,
   and BP is the term that notices.

CORPUS-LEVEL, NOT SENTENCE-AVERAGED. We sum the match counts over the whole
test set and divide once. Averaging per-sentence BLEU is a different (and
much noisier) statistic — a 6-word sentence with no 4-gram match would
contribute a 0 and drag the mean down, even though corpus-level BLEU would
barely notice. When two papers disagree about BLEU, this is usually why.
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Sequence


def ngrams(tokens: Sequence[str], n: int) -> collections.Counter:
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses: List[Sequence[str]], references: List[Sequence[str]],
                max_n: int = 4, smooth: bool = True) -> Dict[str, float]:
    """Corpus BLEU-4. Returns the score and its parts, because the parts are
    where the diagnosis lives.

    `smooth` adds Chen & Cherry (2014) "method 1": add a tiny epsilon to zero
    numerators. Without it, a small test set where the model produced not one
    correct 4-gram reports BLEU exactly 0.0, which is true but useless — you
    cannot tell a model that is close from one that is hopeless. We report both
    so nothing is hidden.
    """
    assert len(hypotheses) == len(references)
    matches = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h_ng = ngrams(hyp, n)
            r_ng = ngrams(ref, n)
            # THE CLIP: credit each n-gram at most as often as the reference has it.
            matches[n] += sum(min(cnt, r_ng[g]) for g, cnt in h_ng.items())
            totals[n] += max(sum(h_ng.values()), 0)

    precisions, raw_precisions = [], []
    for n in range(1, max_n + 1):
        if totals[n] == 0:
            p = 0.0
        else:
            p = matches[n] / totals[n]
        raw_precisions.append(p)
        if p == 0.0 and smooth:
            p = 1.0 / (2 * max(totals[n], 1))     # Chen & Cherry method 1
        precisions.append(p)

    if min(precisions) > 0:
        log_p = sum(math.log(p) for p in precisions) / max_n
        geo = math.exp(log_p)
    else:
        geo = 0.0

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return {
        "bleu": 100 * bp * geo,
        "brevity_penalty": bp,
        "hyp_len": hyp_len,
        "ref_len": ref_len,
        "length_ratio": hyp_len / max(ref_len, 1),
        **{f"p{n}": 100 * raw_precisions[n - 1] for n in range(1, max_n + 1)},
    }


def bleu_by_length(hypotheses: List[Sequence[str]], references: List[Sequence[str]],
                   sources: List[Sequence[str]], bins: Sequence[int] = (0, 5, 10, 15, 20, 25, 30, 40, 999)
                   ) -> List[Dict]:
    """BLEU as a function of SOURCE sentence length — the paper's Figure 3.

    This is the measurement that tests the architecture's central weakness. The
    encoder squeezes every source sentence into the same fixed-size vector, so
    a priori you would expect quality to collapse as sentences get longer. §3.7
    reports that it does not:

        "no degradation on sentences with less than 35 words, there is only a
         minor degradation on the longest sentences"

    We compute corpus BLEU separately within each length bucket. Corpus BLEU
    per bucket (not averaged sentence BLEU) keeps it comparable to the headline
    number.
    """
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, s in enumerate(sources) if lo <= len(s) < hi]
        if len(idx) < 5:            # too few to mean anything; say so, don't hide it
            out.append({"bin": f"{lo}-{hi}", "n": len(idx), "bleu": None})
            continue
        b = corpus_bleu([hypotheses[i] for i in idx], [references[i] for i in idx])
        out.append({"bin": f"{lo}-{hi}", "n": len(idx), "bleu": b["bleu"],
                    "length_ratio": b["length_ratio"]})
    return out
