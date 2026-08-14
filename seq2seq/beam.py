"""
seq2seq.beam — decoding: turning a trained model into translations.

TRAINING AND DECODING ARE DIFFERENT PROBLEMS. During training we had the true
target and could compute all T steps in one shot (teacher forcing). At
inference there is no true target: step t's input is whatever we chose at step
t-1. The model is now consuming its own output distribution, which it never saw
during training — the mismatch known as EXPOSURE BIAS.

Worse, we want the most likely SENTENCE

    argmax_y  p(y | x) = argmax_y  PROD_t p(y_t | v, y_<t)

and that argmax is over an exponentially large set: |V|^T sequences. Exact
search is intractable, so every decoder is an approximation.

    GREEDY      take the argmax at each step. Fast, and wrong whenever the
                best sentence starts with a locally-suboptimal word — which is
                often, because a high-probability first word can lead into a
                low-probability continuation.

    BEAM        keep the B best prefixes at every step instead of 1. Still
                approximate (the best sentence can fall off the beam early),
                but dramatically better than greedy for very little cost.

The paper, §3.2:

    "We search for the most likely translation using a simple left-to-right
     beam search decoder which maintains a small number B of partial
     hypotheses, where a partial hypothesis is a prefix of some translation. At
     each timestep we extend each partial hypothesis in the beam with every
     possible word in the vocabulary. This greatly increases the number of the
     hypotheses so we discard all but the B most likely hypotheses according to
     the model's log probability. As soon as the '<EOS>' symbol is appended to
     a hypothesis, it is removed from the beam and is added to the set of
     complete hypotheses."

and the finding that surprised everyone (§3.2):

    "Our decoder ... works well with a beam size of 1, and a beam of size 2
     provides most of the benefits of beam search."

WHY WE ADD LOG-PROBABILITIES AND NEVER MULTIPLY PROBABILITIES. A 25-word
sentence has probability around 1e-30. In fp32 the smallest normal number is
~1e-38, so a 35-word sentence underflows to exactly 0.0 and every hypothesis
ties at zero. In log space the same sentence is a perfectly ordinary -69, and
multiplication becomes addition. This is not an optimisation, it is the
difference between working and not working.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .data import EOS, PAD, SOS


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, max_len: int = 60) -> List[List[int]]:
    """Take the argmax at every step. The B=1 baseline.

    src: (S, B). Returns B token-id lists, <eos> excluded.
    """
    model.eval()
    device = src.device
    B = src.size(1)
    state = model.encode(src)
    y = torch.full((B,), SOS, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out = [[] for _ in range(B)]

    for _ in range(max_len):
        logp, state = model.decode_step(y, state)      # (B, V)
        y = logp.argmax(dim=-1)
        for b in range(B):
            if not done[b]:
                if y[b].item() == EOS:
                    done[b] = True
                else:
                    out[b].append(y[b].item())
        if bool(done.all()):
            break
    return out


@torch.no_grad()
def beam_search(model, src: torch.Tensor, beam_size: int = 5, max_len: int = 60,
                length_penalty: float = 0.0) -> List[List[int]]:
    """Batched beam search over a whole minibatch of source sentences at once.

    THE INDEX GYMNASTICS, which is the part everybody finds confusing.
    We fold the beam into the batch dimension: the LSTM sees `B*K` "sentences",
    where row `b*K + k` is hypothesis k of sentence b. Everything downstream is
    bookkeeping to keep that mapping straight.

    Per step:
      1. logp (B*K, V) -> view (B, K, V)
      2. add the running score of each hypothesis: (B,K,1) broadcast over V.
         Now cell [b,k,v] is the total log-prob of "prefix k then word v".
      3. flatten to (B, K*V) and take the top K. This is the key move: the
         K survivors may all come from the SAME parent hypothesis, or from K
         different ones. Beam search does not keep one child per parent.
      4. decode the flat index: parent = idx // V, word = idx % V.
      5. REORDER THE LSTM STATE by `parent`. This is the step people forget,
         and forgetting it is silent: you get fluent output that ignores the
         source, because hypothesis k is now being continued with hypothesis
         j's memory.

    FINISHED HYPOTHESES. The paper removes a hypothesis from the beam as soon
    as it emits <eos> and files it under "complete". We do the same by setting
    its score to -inf so it can never be selected again as a parent, after
    recording it. Search stops when every sentence's beam is exhausted or we
    hit max_len.

    LENGTH PENALTY. `length_penalty=0` is the paper: rank by raw total
    log-probability. Note the bias this creates — every extra word adds a
    negative number, so raw log-prob systematically prefers SHORT translations,
    and the effect gets worse as the beam widens (a wider beam is better at
    finding the short high-scoring hypothesis). Dividing by `len^alpha`
    (Wu et al. 2016) corrects it. We default to the paper's behaviour and let
    you turn the correction on to see it.
    """
    model.eval()
    device = src.device
    S, B = src.shape
    K, V = beam_size, model.tgt_vocab

    h, c = model.encode(src)                            # each (L, B, H)
    L, _, H = h.shape
    # (L, B, H) -> (L, B*K, H), where consecutive K rows belong to one sentence.
    h = h.repeat_interleave(K, dim=1).contiguous()
    c = c.repeat_interleave(K, dim=1).contiguous()
    state = (h, c)

    # Only beam 0 starts alive. If all K beams started at score 0 they would
    # all expand identically and the top-K would be K copies of the same word.
    scores = torch.full((B, K), float("-inf"), device=device)
    scores[:, 0] = 0.0
    tokens = torch.full((B, K, 0), 0, dtype=torch.long, device=device)
    y = torch.full((B * K,), SOS, dtype=torch.long, device=device)

    finished: List[List[Tuple[float, List[int]]]] = [[] for _ in range(B)]

    for step in range(max_len):
        logp, state = model.decode_step(y, state)       # (B*K, V)
        logp = logp.view(B, K, V)

        cand = scores.unsqueeze(-1) + logp              # (B, K, V)
        flat = cand.view(B, K * V)
        top_scores, top_idx = flat.topk(K, dim=-1)      # (B, K)
        parent = top_idx // V                           # which hypothesis
        word = top_idx % V                              # which next token

        # Reorder the LSTM state so row (b,k) carries hypothesis `parent[b,k]`'s
        # memory. Miss this and the search is subtly, silently broken.
        gather_idx = (torch.arange(B, device=device).unsqueeze(1) * K + parent).view(-1)
        state = (state[0][:, gather_idx].contiguous(),
                 state[1][:, gather_idx].contiguous())
        tokens = torch.cat([tokens.gather(
            1, parent.unsqueeze(-1).expand(-1, -1, tokens.size(-1))),
            word.unsqueeze(-1)], dim=-1)                # (B, K, step+1)
        scores = top_scores

        # File any hypothesis that just emitted <eos>, and kill it in the beam.
        is_eos = word == EOS
        if bool(is_eos.any()):
            for b, k in is_eos.nonzero(as_tuple=False).tolist():
                seq = tokens[b, k, :-1].tolist()        # drop the <eos> itself
                sc = scores[b, k].item()
                if length_penalty:
                    sc = sc / (max(len(seq), 1) ** length_penalty)
                finished[b].append((sc, seq))
                scores[b, k] = float("-inf")

        if bool(torch.isinf(scores).all()):
            break
        y = word.view(-1)

    out = []
    for b in range(B):
        if finished[b]:
            out.append(max(finished[b], key=lambda t: t[0])[1])
        else:
            # Nothing finished within max_len: fall back to the best live beam.
            k = int(scores[b].argmax())
            out.append([t for t in tokens[b, k].tolist() if t not in (PAD, EOS)])
    return out


@torch.no_grad()
def translate_corpus(model, corpus, pairs, reverse_source: bool, device,
                     beam_size: int = 1, batch_size: int = 64,
                     max_len: int = 60, length_penalty: float = 0.0,
                     sort_by_length: bool = True
                     ) -> Tuple[List[List[str]], List[List[str]]]:
    """Decode a whole split. Returns (hypotheses, references) as token lists.

    `sort_by_length=True` GROUPS SIMILAR-LENGTH SENTENCES INTO THE SAME BATCH,
    then restores the original order before returning. This is not a speed
    optimisation — it is a CORRECTNESS fix, and the bug it fixes is one of the
    nastiest kinds: a train/inference distribution mismatch that we introduced
    ourselves, by optimising training.

    THE BUG, in full, because it is worth more than the fix.

    We train with length bucketing (paper §3.4, "all sentences in a minibatch
    are roughly of the same length"). So during training, batches are nearly
    uniform and the encoder sees almost no padding.

    If you then decode the test set in its natural order, a 6-token sentence
    can land in a batch with a 32-token one. Because we LEFT-pad the source,
    that short sentence's encoder now consumes 26 <pad> embeddings before
    reaching a real word. The <pad> embedding is the zero vector, but an LSTM
    fed zeros does not sit still — its biases keep driving the gates, so the
    hidden state drifts steadily away from the origin for 26 steps. By the time
    the real words arrive, the encoder is starting from a state it has never
    once seen in training, and v comes out wrong.

    The symptom is not a crash. It is fluent, grammatical target-language text
    that ignores the source — the decoder falls back on its language-model
    prior — plus a length ratio far above 1 because a model that never locked
    onto the source also never finds a good place to stop.

    Measured on this model: BLEU 7.64 -> 12.80 and length ratio 1.73 -> 0.94,
    purely from grouping similar lengths together. Same weights, same beam.

    THE MORE PRINCIPLED FIX, for the record: mask the encoder so <pad>
    timesteps genuinely do not update the state (`pack_padded_sequence`, which
    needs right-padding), or carry an explicit length mask through the
    recurrence. That removes the failure mode entirely rather than avoiding it,
    and it is what you should do in production. Sorting is one line, restores
    the exact training distribution, and is what most research code does — but
    know which one you are relying on.
    """
    from . import data as D
    order = list(range(len(pairs)))
    if sort_by_length:
        order.sort(key=lambda i: len(pairs[i][0]))

    hyps_by_idx: dict = {}
    for i in range(0, len(order), batch_size):
        idxs = order[i:i + batch_size]
        chunk = [pairs[j] for j in idxs]
        enc = [D.encode_pair(s, t, corpus, reverse_source) for s, t in chunk]
        src, _, _ = D.pad_batch(enc, device)
        if beam_size <= 1:
            ids = greedy_decode(model, src, max_len)
        else:
            ids = beam_search(model, src, beam_size, max_len, length_penalty)
        for j, seq in zip(idxs, ids):
            hyps_by_idx[j] = corpus.tgt_vocab.decode(seq)

    # Restore the ORIGINAL order. Getting this wrong silently misaligns
    # hypotheses with references and produces a BLEU score that means nothing.
    hyps = [hyps_by_idx[i] for i in range(len(pairs))]
    refs = [list(t) for _, t in pairs]
    return hyps, refs
