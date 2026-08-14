"""
seq2seq.analysis — looking inside v, the fixed-size sentence vector (paper §3.8).

§3.8 is the part of the paper that stopped being about translation and started
being about representation:

    "The figure clearly shows that the representations are sensitive to the
     order of words, while being fairly insensitive to the replacement of an
     active voice with a passive voice."

Figure 2 shows a 2-D PCA of encoder states for a handful of phrases and invites
you to look at the clusters. That is a qualitative argument, and qualitative
arguments about 2-D projections of 1000-dimensional spaces are notoriously easy
to fool yourself with — PCA keeps the two directions of largest variance, which
need not be the two directions carrying the effect you care about.

So we do two things:

  1. reproduce the picture (PCA scatter), because it is the paper's figure, and
  2. replace the eyeballing with a NUMBER, in the full space, with a control.

THE MEASUREMENT. Build matched sentence triples:

    base     "a man in a red shirt is holding a dog"
    swapped  "a dog in a red shirt is holding a man"     <- same words, order changed
    passive  "a dog is being held by a man in a red shirt" <- same meaning, different words

If §3.8 is right, then in encoder space

    distance(base, swapped)  >>  distance(base, passive)

which is the opposite of what a bag-of-words model would say: `swapped` shares
*every token* with `base`, while `passive` shares only some. Any model that
merely averaged word embeddings would rank these the other way round. That
makes the comparison a real test rather than a restatement.

WHICH VECTOR IS "THE" SENTENCE VECTOR? The encoder's final state is 4 layers x
(h, c). The paper's phrase vectors come from the LSTM's hidden state. We use
the TOP LAYER's h by default and expose the alternatives, because the choice
matters and hiding it would be dishonest: lower layers stay closer to lexical
identity, so measuring on layer 0 weakens the effect and measuring on the
concatenation splits the difference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import data as D


@torch.no_grad()
def sentence_vectors(model, corpus, sentences: Sequence[Sequence[str]],
                     reverse_source: bool, device, which: str = "top_h"
                     ) -> torch.Tensor:
    """Encode sentences and return their fixed-size representations, (N, dim).

    `which`:
        "top_h"  the last layer's hidden state   — the paper's phrase vector
        "top_c"  the last layer's cell state
        "all_h"  every layer's h, concatenated   — the full message the decoder
                 actually receives (well, half of it; it also gets all the c's)
    """
    model.eval().to(device)
    enc = [D.encode_pair(s, s, corpus, reverse_source)[0] for s in sentences]
    # Pad manually: pad_batch expects (src, tgt) pairs; we only need sources.
    S = max(len(e) for e in enc)
    src = torch.full((S, len(enc)), D.PAD, dtype=torch.long)
    for b, e in enumerate(enc):
        src[S - len(e):, b] = torch.tensor(e)          # left-pad, as in training
    h, c = model.encode(src.to(device))                # each (L, N, H)
    if which == "top_h":
        return h[-1]
    if which == "top_c":
        return c[-1]
    if which == "all_h":
        return h.permute(1, 0, 2).reshape(h.size(1), -1)
    raise ValueError(which)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


# Probe set. Every content word is chosen to be frequent in Multi30k image
# captions, so that none of them fall out of vocabulary as <unk> — an <unk>
# would make the two members of a pair artificially similar and quietly
# manufacture the result we are testing for.
PROBES: List[Dict[str, str]] = [
    dict(base="a man is holding a dog",
         swapped="a dog is holding a man",
         passive="a dog is being held by a man"),
    dict(base="a woman is watching a boy",
         swapped="a boy is watching a woman",
         passive="a boy is being watched by a woman"),
    dict(base="a girl is pushing a man",
         swapped="a man is pushing a girl",
         passive="a man is being pushed by a girl"),
    dict(base="a boy is carrying a girl",
         swapped="a girl is carrying a boy",
         passive="a girl is being carried by a boy"),
    dict(base="a dog is chasing a child",
         swapped="a child is chasing a dog",
         passive="a child is being chased by a dog"),
    dict(base="a man is pulling a woman",
         swapped="a woman is pulling a man",
         passive="a woman is being pulled by a man"),
]


@torch.no_grad()
def word_order_vs_voice(model, corpus, reverse_source: bool, device,
                        which: str = "top_h", probes=None) -> Dict:
    """The quantitative version of Figure 2.

    Returns per-probe and mean cosine distances for
        base<->swapped  (same words, different meaning)
        base<->passive  (different words, same meaning)
    plus a bag-of-words control so you can see that the ordering is not trivial.
    """
    probes = probes or PROBES
    tok = D.tokenize
    base = [tok(p["base"]) for p in probes]
    swap = [tok(p["swapped"]) for p in probes]
    pasv = [tok(p["passive"]) for p in probes]

    vb = sentence_vectors(model, corpus, base, reverse_source, device, which)
    vs = sentence_vectors(model, corpus, swap, reverse_source, device, which)
    vp = sentence_vectors(model, corpus, pasv, reverse_source, device, which)

    d_swap = cosine_distance(vb, vs)
    d_pasv = cosine_distance(vb, vp)

    # CONTROL: a bag-of-words model, i.e. the average of the source embeddings.
    # For it, `swapped` is IDENTICAL to `base` (distance 0) and `passive` is
    # far. If the LSTM's ordering is the reverse, word order is genuinely
    # encoded in v and not an artefact of which words are present.
    def bow(sents):
        vecs = []
        for s in sents:
            ids = torch.tensor(corpus.src_vocab.encode(s), device=device)
            vecs.append(model.src_embed(ids).mean(0))
        return torch.stack(vecs)

    bb, bs, bp = bow(base), bow(swap), bow(pasv)
    return {
        "which_vector": which,
        "per_probe": [
            {"base": p["base"], "d_word_order": round(ds.item(), 4),
             "d_voice": round(dp.item(), 4)}
            for p, ds, dp in zip(probes, d_swap, d_pasv)],
        "mean_d_word_order": round(d_swap.mean().item(), 4),
        "mean_d_voice": round(d_pasv.mean().item(), 4),
        "ratio_order_over_voice": round((d_swap.mean() / d_pasv.mean()).item(), 3),
        "bow_control_d_word_order": round(cosine_distance(bb, bs).mean().item(), 4),
        "bow_control_d_voice": round(cosine_distance(bb, bp).mean().item(), 4),
    }


def pca_2d(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain PCA via SVD, so nothing is hidden behind sklearn.

    Centre the data, take the SVD, project onto the top two right-singular
    vectors. `explained` is the fraction of total variance those two directions
    carry — always report it, because a 2-D picture of a 512-D space that keeps
    9% of the variance is a decoration, not evidence.
    """
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc.double(), full_matrices=False)
    coords = Xc.double() @ Vh[:2].T
    explained = (S[:2] ** 2).sum() / (S ** 2).sum()
    return coords.float(), explained.float()
