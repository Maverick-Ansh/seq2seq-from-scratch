# The atoms

Every concept the notebook uses, defined once, in the order you meet them. If a
term in the notebook feels like it was assumed rather than explained, it is
here.

This is written for someone who can program but has not built a neural network
from parts. Nothing is defined in terms of something further down.

---

## Part 1 — the substrate

**Tensor.** An n-dimensional array of numbers plus a *shape*. A scalar is shape
`()`, a vector `(5,)`, a matrix `(3, 4)`. Almost every bug you will hit in deep
learning is a shape bug, so read shapes obsessively. In this repo every sequence
tensor is **time-first**: `(T, B, H)` = timesteps × batch × features. When a
tensor surprises you, check that first.

**Batch.** Instead of processing one example, process `B` of them stacked into
one tensor. Not merely a speed trick: the gradient from a batch is an *average*
over `B` examples, so it is a less noisy estimate of the true gradient than any
single example gives. `B` is therefore both a performance knob and a
statistical one.

**GPU.** A processor with thousands of small cores that all run the same
instruction on different data. A matrix multiply is exactly that shape of
problem, which is why deep learning runs on GPUs. Two consequences that matter
constantly:
- **A GPU is only fast when it is given big, uniform work.** A batch of 16 does
  not use a T4 any better than a batch of 128 does — you pay the same launch
  latency for a fraction of the work. This is why Chapter 4's pipelining stops
  helping past 2 micro-batches.
- **GPU memory is separate from system RAM**, and moving between them (or
  between two GPUs) is slow relative to compute. Most parallelism engineering is
  about *not moving things*.

**CUDA asynchrony.** When you write `y = x @ w`, Python does not wait for the
result — it *queues* the work on the GPU and returns immediately. This is why:
- timing GPU code without `torch.cuda.synchronize()` measures how fast Python
  can queue, not how fast the GPU computes;
- calling `.item()` or `.cpu()` inside a loop is a performance disaster: it
  forces the CPU to stop and wait, destroying any overlap you had.

**fp32 / fp16 / bf16.** How many bits per number. fp32 (4 bytes) is the default.
fp16 (2 bytes) halves memory and, on tensor-core hardware, multiplies throughput
several-fold — but has a narrow range, so values above ~65,504 become `inf`.
bf16 trades precision for fp32's range and is generally easier, **but a Tesla T4
(compute capability 7.5) has no bf16 hardware** — asking for it gets you slow
emulation. On this hardware: fp16 or fp32.

---

## Part 2 — learning

**Parameter / weight.** A number the model adjusts during training. "384M
parameters" means 384 million such numbers.

**Forward pass.** Run input → output through the model.

**Loss.** One number saying how wrong the output was. Training = making it
smaller.

**Gradient.** For each parameter, the derivative of the loss with respect to it:
"if I nudge this weight up slightly, does the loss go up or down, and how
fast?" The gradient of all parameters together is a vector pointing in the
direction of steepest loss *increase*, so we step the other way.

**Backward pass / backpropagation.** Computing all those derivatives by applying
the chain rule from the loss backwards through the network. Costs about twice a
forward pass, and needs the forward pass's intermediate values kept in memory —
which is what "activation memory" means in Chapter 4.

**Autograd.** PyTorch recording every operation you perform into a graph, so it
can run the chain rule for you. `loss.backward()` walks that graph. `.detach()`
cuts a tensor out of it — occasionally what you want, and a silent disaster when
it isn't.

**SGD (stochastic gradient descent).** `weight ← weight − lr × gradient`. That
is the whole algorithm. "Stochastic" because each step uses a random batch
rather than the full dataset.

**Learning rate (`lr`).** How big a step to take. The single most important
hyperparameter. Too large diverges; too small crawls. **`lr` is only meaningful
relative to the scale of your gradient** — which is why the paper's `0.7` and a
modern Adam's `3e-4` are not comparable numbers, and why Chapter 5's
normalisation bug mattered so much.

**Momentum.** Averaging the gradient over recent steps to smooth out noise.
Costs one extra copy of every parameter in memory. **The paper deliberately does
not use it** (§3.4), which on a 384M model saves 1.5 GB.

**Adam.** An optimiser that keeps a running estimate of each parameter's typical
gradient magnitude and divides by it, making the step size roughly scale-free.
Costs *two* extra copies of every parameter. Almost always better than SGD
out-of-the-box; the 2014 paper predates its widespread use.

**Gradient clipping.** If the gradient vector's length exceeds a threshold,
scale the whole vector down to that length. Preserves *direction*, caps
*step size*. Stops one freak batch from destroying the model, which is what
makes a large learning rate survivable. Note: **by global norm**, not
per-coordinate — per-coordinate clipping changes the direction and is a
different, worse algorithm.

**Epoch.** One pass over the training set. **Overfitting**: the model memorising
training examples instead of learning the pattern — visible as training loss
falling while validation loss rises.

**Train / validation / test.** Three disjoint splits. You fit on train, choose
hyperparameters on validation, and touch test exactly once at the end. If you
tune anything against test, your test number is no longer an estimate of
anything.

---

## Part 3 — sequences

**Token.** The unit of text the model sees — here, a word or a punctuation mark.

**Vocabulary.** The fixed list of tokens the model knows, each mapped to an
integer id. Fixed because the output layer's width *is* the vocabulary size.

**`<unk>`, `<pad>`, `<sos>`, `<eos>`.** Special tokens. `<unk>` replaces any word
outside the vocabulary. `<pad>` fills short sentences so a batch is rectangular.
`<sos>` starts the decoder. **`<eos>` is how the model says "I'm done"** — the
paper's §2 notes this is what "enables the model to define a distribution over
sequences of all possible lengths".

**Embedding.** A lookup table mapping each token id to a learned vector. Turns
discrete symbols into something you can do arithmetic on.

**RNN (recurrent neural network).** A network with a loop: it keeps a *hidden
state* and updates it once per token. That state is its memory of everything so
far.

**BPTT (backpropagation through time).** Backprop applied to an RNN by
"unrolling" the loop into a T-step-deep feed-forward network. This is why a
30-token sentence is effectively a 30-layer network, and why gradients have so
far to travel.

**Vanishing / exploding gradients.** In a T-step unroll, gradient is multiplied
by a similar factor T times. Below 1 it decays to nothing; above 1 it blows up.
The reason plain RNNs cannot learn long-range dependencies.

**LSTM.** An RNN whose state has two parts — a cell state `c` (long-term) and a
hidden state `h` (exposed) — and three gates controlling what is written,
what is kept, and what is revealed. The critical property: `∂c_t/∂c_{t-1} = f_t`
is an elementwise multiply with **no weight matrix in the path**, so if the
model learns `f ≈ 1` the gradient flows back undamped. Chapter 1 measures this.

**Teacher forcing.** At training time, feed the decoder the *true* previous word
rather than its own previous prediction. Lets all T steps be computed in one
parallel pass and stops early mistakes cascading.

**Exposure bias.** The cost of teacher forcing: at inference the model must
consume its own outputs, a distribution it never saw in training.

**Softmax.** Turns a vector of scores ("logits") into a probability
distribution. Here it runs over the entire target vocabulary at every output
position — which is why it is the most expensive operation in the model.

**Cross-entropy loss.** `−log(probability the model assigned to the correct
token)`. Zero if the model was certain and right; large if it was confident and
wrong.

**Perplexity.** `exp(mean per-token cross-entropy)`. Read it as "how many words
is the model effectively choosing between at each step". Perplexity 1 = knows
exactly; perplexity = vocabulary size = knows nothing. The paper's §3.3 reports
5.8 → 4.7 from reversing the source.

**Greedy decoding / beam search.** At inference you want the most likely
*sentence*, but there are `|V|^T` of them. Greedy takes the best token at each
step and can be trapped by a locally-good first word. Beam search keeps the `B`
best prefixes at each step. Both are approximations; beam is a much better one.

**BLEU.** The translation metric: what fraction of the model's n-grams
(n = 1..4) appear in the reference, with each n-gram credited at most as often
as the reference contains it, combined by geometric mean, times a penalty for
being too short.

---

## Part 4 — many GPUs

**Data parallelism.** Every GPU holds a **full copy** of the model; you split
the **batch**. After backward, gradients are averaged across copies
(*all-reduce*) so all copies stay identical. Communication is
`O(parameters)` per step. Requires the model to fit on one GPU.

**Model / tensor parallelism.** Each GPU holds a **different piece** of the
model; every GPU sees every example. Communication is `O(batch × seq × hidden)`
— the *activations* at each cut point. Solves "doesn't fit", and (as Chapter 4
finds) sometimes wins on speed too, when parameters vastly outnumber
activations.

**Pipeline parallelism.** Model parallelism plus splitting the batch into
**micro-batches**, so stage *k* works on micro-batch *i* while stage *k+1* works
on *i−1*.

**The bubble.** With naive model parallelism, stage 2 cannot start until stage 1
finishes, so each GPU idles ~50% of the time. With `S` stages and `M`
micro-batches, utilisation is `M / (M + S − 1)` — the GPipe formula. **It is an
upper bound**: it assumes per-device efficiency is unchanged, and shrinking
micro-batches makes that false.

**All-reduce.** A collective operation where every participant ends up with the
sum (or average) of everyone's values. NCCL implements it as a bandwidth-optimal
ring.

**NCCL.** NVIDIA's GPU-to-GPU collective communication library. The backend
behind `DistributedDataParallel`.

**P2P (peer-to-peer).** Whether GPU A can DMA directly into GPU B's memory
without staging through system RAM. Chapter 0 checks this; without it, every
model-parallel layer boundary costs roughly double.

**`nn.DataParallel` vs `DistributedDataParallel`.** `DataParallel` is one
process with threads, and it **re-replicates the entire model to the other GPU
on every forward pass**. Measured in Chapter 4: 1.00× speedup, i.e. none. `DDP`
is one process per GPU with the model resident, all-reducing gradients inside
`backward()` where the communication overlaps compute. **Use DDP. Never use
`DataParallel`.**

---

## Part 5 — the paper's own vocabulary

**Seq2seq / encoder-decoder.** Read the input with one network into a fixed-size
vector; generate the output with a second network conditioned on that vector.

**The bottleneck.** Everything the decoder learns about the source must pass
through that one vector — 8000 numbers in the paper's model, *regardless of
source length*. The architecture's central weakness, and the thing attention
later removed.

**Attention.** (Not in this paper — Bahdanau et al., months later.) Let the
decoder look back at *every* encoder timestep, weighted by relevance, instead of
only the final state. Removes the bottleneck. Three years after that,
"attention is all you need" removed the recurrence too.

**The reversal trick (§3.3).** Feed the source sentence backwards. Leaves the
*average* source-to-target distance unchanged but collapses the *minimum* from
T to ~0, which is what optimisation needs to bootstrap. Worth +4.7 BLEU in the
paper, for free.
