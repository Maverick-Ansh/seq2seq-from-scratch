# Sequence to Sequence Learning with Neural Networks — from scratch, on 2×T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/seq2seq-from-scratch/blob/main/notebooks/seq2seq_from_scratch.ipynb)

**→ [Read the notebook](notebooks/seq2seq_from_scratch.ipynb)** (53 cells, half of them prose) · **[Glossary of every concept used](GLOSSARY.md)** · **[All result numbers](artifacts/)**

A line-by-line reproduction of **Sutskever, Vinyals & Le (2014)**, [arXiv:1409.3215](https://arxiv.org/abs/1409.3215), built for someone who has never assembled a neural network from parts.

Every component is written from its equations — the LSTM cell, the encoder-decoder, the training loop, beam search, BLEU, and the multi-GPU parallelism — and every claim in the paper is **measured rather than quoted**, including the ones that don't reproduce.

Hardware: **Kaggle 2× Tesla T4**, verified in Chapter 0.

---

## What the paper did, in one sentence

In 2014, machine translation was a pipeline of alignment models, phrase tables and tuned rerankers. This paper replaced the whole thing with **two LSTMs and a loop** — read the source into one fixed-size vector, unroll the target from it — and beat the pipeline. Everything since (attention, the Transformer, GPT) descends from this idea.

---

## Results

### Reproduced

| Paper's claim | Paper | This repo | |
|---|---|---|---|
| Parameter count of the §3.4 model | 384M (64M recurrent) | **384,144,000** (64,064,000) | ✅ exact |
| Length bucketing speedup (§3.4) | "a 2× speedup" | **2.00×** fewer sequential timesteps | ✅ exact |
| **Reversing the source** (§3.3) | perplexity −19.0% | **−18.3%** (3 seeds/arm, gap = 5.2σ) | ✅ |
| Beam width (§3.2) | "a beam of size 2 provides most of the benefits" | beam 2 captures **100%** of the gain; beam 12 costs 13× the time for −0.02 BLEU | ✅ |
| Hand-written LSTM ≡ `nn.LSTM` | — | agree to **1e-16** fwd + bwd (float64) | ✅ |
| Vocab-sharded softmax is exact | — | loss and gradients match to **1e-16** | ✅ |

### Did not reproduce (and why)

| Paper's claim | Paper | This repo |
|---|---|---|
| No degradation on long sentences (§3.7) | flat below 35 words | **−27%** BLEU for sources ≥20 tokens |
| $v$ is more order- than voice-sensitive (§3.8) | qualitative (Fig 2) | ratio **0.75** (want >1), consistent across 3 seeds |

Chapter 8.4 tests our own explanation for both — that our bottleneck is too small — by sweeping the hidden size and re-measuring. **The prediction was refuted:**

| hidden | v size | valid ppl | BLEU | BLEU short | BLEU long | "degradation" |
|---|---|---|---|---|---|---|
| 256 | 2,048 | 18.59 | 6.11 | 5.41 | 5.47 | **+1.1%** |
| 512 | 4,096 | 11.01 | 11.10 | 11.51 | 6.90 | −40.1% |
| 768 | 6,144 | 8.35 | **18.09** | 19.08 | 10.69 | −44.0% |

A *smaller* bottleneck showed *less* degradation, not more — because the 256-unit model is uniformly terrible (BLEU ~5.4 on everything) and a relative gap is meaningless when the baseline is on the floor. The remaining candidate is data sparsity in the long tail, which we have **not** tested and do not claim.

(Incidental finding: our headline model was undersized. 512 → 768 units takes BLEU from 11.10 to 18.09.)

### The multi-GPU result

The paper's exact 384M model, batch 128, on 2× T4:

| regime | ms/step | words/s | vs 1 GPU |
|---|---|---|---|
| 1 GPU | 1141 | 6,728 | 1.00× |
| 2 GPU model-parallel, no pipeline | 727 | 10,563 | 1.57× |
| **2 GPU model-parallel, pipeline ×2** | **648** | **11,852** | **1.76×** |
| 2 GPU model-parallel, pipeline ×4 | 774 | 9,928 | 1.48× |
| 2 GPU model-parallel, pipeline ×8 | 984 | 7,805 | 1.16× |
| 2 GPU `nn.DataParallel` | 1144 | 6,712 | **1.00×** |
| 2 GPU real DDP (`torchrun`) | 875 | 8,773 | 1.30× |
| *paper, 8 GPUs (2014)* | | *6,300* | *3.71×* |

Four things worth knowing, all measured here:

1. **One free T4 beats the paper's entire 8-GPU machine** (7,250 vs 6,300 words/sec). The algorithm didn't change; the silicon did.
2. **`nn.DataParallel` gives exactly zero speedup** — it re-replicates 1.54 GB of model to the second GPU on *every forward pass*. Never use it.
3. **Model parallelism beat data parallelism here**, which is the opposite of the usual advice — because it also shards the softmax, and because moving activations (61 MB) beats moving gradients (1,540 MB) when parameters dwarf activations.
4. **Pipelining peaks at 2 micro-batches and then *hurts*.** The GPipe bubble formula $\frac{M}{M+S-1}$ is an upper bound that assumes per-device efficiency is unchanged; shrinking micro-batches makes that false.

---

## Two bugs this repo found by measurement

Both are documented in full in the notebook, because the debugging method matters more than the fixes.

**1. A hyperparameter that silently did nothing.** The paper's gradient clip threshold of 5 assumes a *per-sentence* loss normalisation. Under the modern *per-token* convention our gradients were ~10× smaller, so `clip=5` fired on **0.0%** of batches and `lr=0.7` under-stepped — a training script that *said* "clipping as in §3.4" and did nothing at all. Fixing the normalisation: perplexity **141.5 → 56.1**.

> Habit: for every hyperparameter you set, log the quantity it controls. Silent no-ops are the most expensive bugs in ML because nothing ever fails.

**2. A train/inference mismatch we introduced ourselves.** Training uses length bucketing (§3.4), so the encoder almost never sees padding. Decoding the test set in natural order put 6-token sentences in a batch with 32-token ones; with left-padding, the encoder then consumed 26 `<pad>` steps whose zero embeddings still drive the gates through the biases. Symptom: fluent French that ignored the English, length ratio 1.73. Fix: **BLEU 6.42 → 11.10**, length ratio → 1.02, same weights.

> Every training-time optimisation is also a change to the training distribution. Ask of each one: *what does this stop the model from ever seeing?*

---

## Repo layout

```
seq2seq/
  lstm.py       the four LSTM equations, a deep stack, and the proof it equals nn.LSTM
                + grad_flow_demo: the vanishing-gradient measurement
  data.py       Multi30k, closed vocabulary, the reversal trick, length bucketing
  model.py      encoder → v → decoder (Eq. 1), paper init, ensembling
  train.py      §3.4 recipe: SGD 0.7, no momentum, halving schedule, global-norm clip
  parallel.py   layer-split model parallelism + vocab-parallel softmax + pipelining
  beam.py       greedy and batched beam search, written out index by index
  bleu.py       modified n-gram precision, brevity penalty, BLEU-by-length
  analysis.py   §3.8 sentence-vector probes, made quantitative
scripts/
  ddp_bench.py  real DistributedDataParallel benchmark (torchrun)
GLOSSARY.md     every concept used, defined once, for a complete beginner
artifacts/      figures and result JSON from the run
```

---

## The chapters

| | | paper |
|---|---|---|
| **0** | The machine; why sequences break ordinary neural nets | §1 |
| **1** | An LSTM from its four equations, proven equal to `nn.LSTM`; the gradient highway, measured | — |
| **2** | Sentences → tensors: closed vocabulary, `<eos>`, the reversal, length bucketing | §3.1, §3.4 |
| **3** | Encoder → $v$ → decoder; parameter accounting; the bottleneck | §2, Eq. 1 |
| **4** | **Splitting the model across 2 GPUs** — memory budgets, three parallelisms, sharded softmax, the bubble | §3.5 |
| **5** | The training recipe, and the normalisation trap | §3.4 |
| **6** | The reversal trick as a controlled A/B, 3 seeds per arm | §3.3 |
| **7** | Beam search, BLEU, and finding a real bug | §3.2, §3.6 |
| **8** | Ensembling, long sentences, and what's inside $v$ | §3.6–3.8 |

---

## Scale, stated honestly

| | paper | this repo |
|---|---|---|
| corpus | WMT'14 En→Fr | Multi30k En→Fr |
| sentence pairs | 12,000,000 | 29,000 |
| vocabulary | 160k / 80k | 5.9k / 6.4k |
| model | 4×1000, 384M params | 4×512, 26M params |
| training | 8 GPUs, 10 days | 2× T4, ~16 min for 6 runs |

**400× less data.** Absolute BLEU is therefore far below the paper's 34.8, and should be. What reproduces here are the *mechanisms* and the *effect sizes* — which is the right thing to reproduce, because a claim about optimisation geometry doesn't care how big your corpus is.

---

## Running it

The notebook clones this repo onto the Kaggle/Colab backend and imports from it:

```python
!git clone https://github.com/Maverick-Ansh/seq2seq-from-scratch
import sys; sys.path.insert(0, "seq2seq-from-scratch")
```

Needs 2 GPUs only for Chapter 4 and the concurrent A/B runs; everything else works on one.

---

## Citation

```bibtex
@inproceedings{sutskever2014sequence,
  title={Sequence to Sequence Learning with Neural Networks},
  author={Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V},
  booktitle={NeurIPS},
  year={2014}
}
```
