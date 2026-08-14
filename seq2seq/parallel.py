"""
seq2seq.parallel — putting one model on two GPUs, the way the paper did it.

THE PAPER'S SETUP (§3.5), quoted in full because every clause is a decision:

    "A C++ implementation of deep LSTM with the configuration from the previous
     section on a single GPU processes a speed of approximately 1,700 words per
     second. This was too slow for our purposes, so we parallelized our model
     using an 8-GPU machine. Each layer of the LSTM was executed on a different
     GPU and communicated its activations to the next GPU / layer as soon as
     they were computed. Our models have 4 layers of LSTMs, each of which
     resides on a separate GPU. The remaining 4 GPUs were used to parallelize
     the softmax, so each GPU was responsible for multiplying by a
     1000 x 20000 matrix. The resulting implementation achieved a speed of
     6,300 (both English and French) words per second with a minibatch size of
     128. Training took about a ten days with this implementation."

Unpack it:
  * 4 GPUs hold ONE LAYER EACH        -> LAYER-WISE MODEL PARALLELISM
  * 4 GPUs hold a SLICE OF THE SOFTMAX -> TENSOR PARALLELISM over the vocab
    axis (80,000 / 4 = 20,000 columns each — exactly the "1000 x 20000")
  * "as soon as they were computed"   -> PIPELINING: layer k+1 starts on
    timestep t while layer k works on timestep t+1
  * 1,700 -> 6,300 words/sec = 3.7x from 8 GPUs. NOT 8x. Understanding why
    that number is 3.7 and not 8 is most of what this module teaches.

WE HAVE TWO GPUs, so we map the same ideas down:
  * 4 LSTM layers -> 2 per GPU  (cuda:0 gets layers 0-1, cuda:1 gets 2-3)
  * the softmax   -> split the vocabulary in half, one half per GPU
  * pipelining    -> micro-batches, so both GPUs are busy at once

THE THREE PARALLELISMS — know which problem each one solves
-----------------------------------------------------------
DATA PARALLELISM        Every GPU holds a FULL COPY of the model; you split the
                        BATCH. After backward, all-reduce the gradients so every
                        copy stays identical.
                        Solves: not enough throughput.
                        Requires: the model FITS on one GPU.
                        Cost: communicating gradients = O(parameters) per step.

MODEL / TENSOR PAR.     Each GPU holds a DIFFERENT PIECE of the model; every
                        GPU sees every example.
                        Solves: the model does not fit on one GPU.
                        Cost: communicating ACTIVATIONS at each cut point,
                        = O(batch x seq x hidden) per step.

PIPELINE PARALLELISM    Model parallelism + splitting the batch into
                        micro-batches so stage k works on micro-batch i while
                        stage k+1 works on micro-batch i-1.
                        Solves: the IDLE BUBBLE that naive model parallelism
                        creates (with 2 stages and no pipelining, each GPU is
                        idle ~50% of the time).

The 2014 paper needed model parallelism because 384M parameters did not fit in
a 2014 GPU (a K40 had 12 GB, and the activations for the 80k-way softmax alone
are enormous). If your model fits, data parallelism is almost always faster —
gradients all-reduce once per step, whereas model parallelism ships activations
at every layer boundary at every timestep.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD
from .model import Seq2Seq

State = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Memory accounting — know what actually lives on the card
# ---------------------------------------------------------------------------

def memory_budget(n_params: int, optimizer: str = "sgd", dtype_bytes: int = 4) -> dict:
    """What a training step costs in GPU memory, before activations.

    Four things live on the card during training, and beginners usually count
    only the first:

      1. WEIGHTS            n_params x 4 bytes (fp32)
      2. GRADIENTS          the same size again — one number per weight
      3. OPTIMIZER STATE    SGD without momentum: ZERO. SGD+momentum: 1x.
                            Adam: 2x (exp_avg and exp_avg_sq).
      4. ACTIVATIONS        everything the backward pass needs to recompute
                            gradients. Scales with batch x sequence length, NOT
                            with parameter count, and for an RNN it is
                            proportional to the number of TIMESTEPS.

    The paper uses "stochastic gradient descent without momentum" (§3.4). On a
    384M-parameter model that choice alone saves 1.5 GB versus momentum and
    3.1 GB versus Adam. In 2014, on 12 GB cards, that was not a stylistic
    preference.
    """
    mult = {"sgd": 0, "momentum": 1, "adam": 2}[optimizer]
    w = n_params * dtype_bytes
    return {
        "weights_GB": w / 1e9,
        "gradients_GB": w / 1e9,
        "optimizer_state_GB": w * mult / 1e9,
        "subtotal_GB": w * (2 + mult) / 1e9,
    }


def activation_estimate(batch: int, seq_len: int, hidden: int, layers: int,
                        vocab: int, dtype_bytes: int = 4) -> dict:
    """Rough activation memory for one forward pass, split by where it goes.

    THE POINT OF THIS FUNCTION: for a small vocabulary the LSTM dominates; for
    the paper's 80,000-word vocabulary the LOGITS dominate everything else by
    an order of magnitude. That single fact explains why half the paper's GPUs
    are assigned to the softmax.
    """
    # Per layer per timestep an LSTM must keep the 4 gate activations plus c
    # and h to backprop through: ~6 tensors of (B, H).
    lstm = layers * seq_len * batch * hidden * 6 * dtype_bytes
    # The logits tensor (T, B, V), plus log-softmax output of the same size.
    logits = seq_len * batch * vocab * 2 * dtype_bytes
    return {"lstm_activations_GB": lstm / 1e9,
            "logits_GB": logits / 1e9,
            "total_GB": (lstm + logits) / 1e9,
            "logits_share": round(logits / (lstm + logits), 3)}


def gpu_mem(device) -> dict:
    return {"allocated_GB": torch.cuda.memory_allocated(device) / 1e9,
            "reserved_GB": torch.cuda.memory_reserved(device) / 1e9,
            "peak_GB": torch.cuda.max_memory_allocated(device) / 1e9}


# ---------------------------------------------------------------------------
# Vocabulary-parallel cross entropy — the paper's sharded softmax, done right
# ---------------------------------------------------------------------------

def vocab_parallel_cross_entropy(shard_logits: List[torch.Tensor],
                                 target: torch.Tensor,
                                 offsets: List[int],
                                 gather_device: torch.device) -> torch.Tensor:
    """Cross-entropy when the vocabulary axis is split across devices.

    THE NAIVE WAY, and why it is bad. Move every shard's logits to one GPU,
    `torch.cat` them, call cross_entropy. That ships a (T, B, V) tensor across
    PCIe. For the paper's numbers — T=30, B=128, V=80,000, fp32 — that is
    **1.2 GB per step, per shard**. PCIe 3.0 x16 gives ~12 GB/s, so you would
    spend ~100 ms per step just moving logits. The whole point of sharding
    would be undone by the gather.

    THE RIGHT WAY. Cross-entropy only needs two numbers per position:

        loss = -( z_target  -  logsumexp(z) )

    and both can be computed with O(T*B) communication instead of O(T*B*V):

      1. each shard computes its own max over its slice          -> (T,B)
      2. all-reduce max  (here: 2 tensors, take elementwise max) -> (T,B)
      3. each shard computes sum(exp(z - global_max)) over slice -> (T,B)
      4. all-reduce sum                                          -> (T,B)
      5. logsumexp = global_max + log(global_sum)
      6. the shard that OWNS the target id contributes z_target; the others
         contribute 0. Sum them.                                 -> (T,B)

    Communication drops from 1.2 GB to about 120 KB. Same number, 10,000x less
    traffic. This is exactly the trick Megatron-LM uses for its vocab-parallel
    output layer, and it is a good example of a general principle: **do not
    move a tensor to compute a reduction over it — move the reduction.**

    Subtracting the global max before exponentiating is not optional. Logits
    routinely reach +30; exp(30) = 1e13 is fine in fp32 but exp(90) overflows to
    inf and the loss becomes NaN. Subtracting the max makes the largest
    exponent exactly 0 and is algebraically free.
    """
    T, B = target.shape
    flat_t = target.reshape(-1)                                   # (T*B,)

    # 1-2. global max over the vocabulary axis
    local_max = [sl.reshape(-1, sl.size(-1)).max(dim=-1).values.to(gather_device)
                 for sl in shard_logits]                           # each (T*B,)
    global_max = local_max[0]
    for m in local_max[1:]:
        global_max = torch.maximum(global_max, m)

    # 3-4. global sum of exp(z - max)
    total_exp = torch.zeros_like(global_max)
    for sl in shard_logits:
        gm = global_max.to(sl.device)
        total_exp = total_exp + (sl.reshape(-1, sl.size(-1)) - gm.unsqueeze(1)
                                 ).exp().sum(dim=-1).to(gather_device)
    log_z = global_max + total_exp.log()                           # (T*B,)

    # 6. gather the target's own logit from whichever shard owns it
    z_target = torch.zeros_like(log_z)
    for sl, off in zip(shard_logits, offsets):
        V_shard = sl.size(-1)
        local_id = flat_t.to(sl.device) - off
        owned = (local_id >= 0) & (local_id < V_shard)
        safe = local_id.clamp(0, V_shard - 1)
        picked = sl.reshape(-1, V_shard).gather(1, safe.unsqueeze(1)).squeeze(1)
        z_target = z_target + (picked * owned).to(gather_device)

    nll = log_z - z_target                                         # (T*B,)
    mask = (flat_t.to(gather_device) != PAD).float()
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# The model-parallel seq2seq
# ---------------------------------------------------------------------------

class ModelParallelSeq2Seq(nn.Module):
    """The paper's §3.5 layout, mapped onto however many GPUs you have.

    LAYER SPLIT. With L=4 layers and 2 devices, layers 0-1 live on cuda:0 and
    layers 2-3 on cuda:1. Encoder and decoder are split the same way, so the
    encoder's final state for layers 0-1 is already sitting on the device that
    needs it as the decoder's initial state for layers 0-1 — no transfer.

    WHAT CROSSES THE PCIe BUS, per forward pass:
      * the layer-1 -> layer-2 activations, a (S, B, H) tensor for the encoder
        and (T, B, H) for the decoder;
      * the decoder's final hidden states, (T, B, H), sent to device 0 so that
        device 0's vocabulary shard can be computed;
      * a handful of (T, B) reduction scalars for the sharded softmax.
    Note what does NOT cross: the (T, B, V) logits. See
    `vocab_parallel_cross_entropy`.

    VOCAB SPLIT. The generator's weight matrix (H, V) is cut along V. Device d
    holds columns [offsets[d], offsets[d] + shard). The paper cuts 80,000 into
    4 x 20,000; we cut it into 2 halves.
    """

    def __init__(self, src_vocab: int, tgt_vocab: int, emb_dim: int = 512,
                 hidden_dim: int = 512, num_layers: int = 4, dropout: float = 0.0,
                 devices: Optional[List[torch.device]] = None,
                 init_range: float = 0.08):
        super().__init__()
        if devices is None:
            devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.devices = devices
        self.n_dev = len(devices)
        assert num_layers % self.n_dev == 0, "layers must divide evenly across devices"
        self.per_dev = num_layers // self.n_dev
        self.num_layers, self.hidden_dim = num_layers, hidden_dim
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        d0 = devices[0]
        # Embeddings sit on device 0: they are pure memory, no compute worth
        # splitting, and the input tokens arrive there anyway.
        self.src_embed = nn.Embedding(src_vocab, emb_dim, padding_idx=PAD).to(d0)
        self.tgt_embed = nn.Embedding(tgt_vocab, emb_dim, padding_idx=PAD).to(d0)

        # One LSTM *stage* per device. Stage 0 takes embeddings (emb_dim in),
        # later stages take the previous stage's hidden output (hidden_dim in).
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i, dev in enumerate(devices):
            in_dim = emb_dim if i == 0 else hidden_dim
            self.enc_stages.append(
                nn.LSTM(in_dim, hidden_dim, self.per_dev, dropout=dropout).to(dev))
            self.dec_stages.append(
                nn.LSTM(in_dim, hidden_dim, self.per_dev, dropout=dropout).to(dev))

        # Sharded generator: device d owns a contiguous slice of the vocabulary.
        base, rem = divmod(tgt_vocab, self.n_dev)
        self.shard_sizes = [base + (1 if i < rem else 0) for i in range(self.n_dev)]
        self.offsets = [sum(self.shard_sizes[:i]) for i in range(self.n_dev)]
        self.gen_shards = nn.ModuleList([
            nn.Linear(hidden_dim, self.shard_sizes[i]).to(devices[i])
            for i in range(self.n_dev)])

        for p in self.parameters():
            nn.init.uniform_(p, -init_range, init_range)
        with torch.no_grad():
            self.src_embed.weight[PAD].zero_()
            self.tgt_embed.weight[PAD].zero_()

    # -- forward ------------------------------------------------------------

    def _run_stages(self, stages, x, states: Optional[List[State]] = None):
        """Push a sequence through the stage chain, moving it between devices.

        `x.to(dev, non_blocking=True)` is the actual inter-GPU hop. With P2P
        enabled (we checked in Chapter 0) this is a direct DMA between the two
        cards' memories; without it, PyTorch stages the copy through pinned
        host memory and it costs roughly double.

        The copy is differentiable — autograd records it and the backward pass
        sends the gradient back the other way across the same link. So the
        traffic figures below apply in both directions.
        """
        out_states: List[State] = []
        h = x
        for i, (stage, dev) in enumerate(zip(stages, self.devices)):
            if h.device != dev:
                h = h.to(dev, non_blocking=True)
            st = None
            if states is not None:
                st = (states[i][0].to(dev, non_blocking=True),
                      states[i][1].to(dev, non_blocking=True))
            h, s = stage(h, st)
            out_states.append(s)
        return h, out_states

    def encode(self, src: torch.Tensor):
        emb = self.src_embed(src.to(self.devices[0]))
        _, states = self._run_stages(self.enc_stages, emb)
        return states                       # one (h, c) per stage, on its device

    def decode_hidden(self, tgt_in: torch.Tensor, states):
        emb = self.tgt_embed(tgt_in.to(self.devices[0]))
        h, _ = self._run_stages(self.dec_stages, emb, states)
        return h                            # (T, B, H) on the LAST device

    def generator_shards(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        """Apply each vocabulary shard on its own device.

        We ship the (T, B, H) hidden states to each device rather than shipping
        (T, B, V) logits back. H=512 vs V=80,000: a 150x smaller message.
        """
        return [shard(hidden.to(dev, non_blocking=True))
                for shard, dev in zip(self.gen_shards, self.devices)]

    def forward(self, src, tgt_in):
        return self.generator_shards(self.decode_hidden(tgt_in, self.encode(src)))

    def loss(self, src, tgt_in, tgt_out):
        shards = self.forward(src, tgt_in)
        return vocab_parallel_cross_entropy(
            shards, tgt_out.to(self.devices[0]), self.offsets, self.devices[0])

    # -- pipelined variant --------------------------------------------------

    def loss_pipelined(self, src, tgt_in, tgt_out, chunks: int = 4):
        """Same computation, but split the BATCH into micro-batches.

        THE BUBBLE. With 2 stages and one monolithic batch, the timeline is:

            cuda:0  [==== stage0 ====]                  (idle)
            cuda:1                    [==== stage1 ====]

        Each GPU is idle half the time. Two GPUs, one GPU's worth of work.

        With 4 micro-batches:

            cuda:0  [s0 m1][s0 m2][s0 m3][s0 m4]
            cuda:1         [s1 m1][s1 m2][s1 m3][s1 m4]

        The stages overlap for all but the first and last micro-batch, so
        utilisation goes from 1/2 to 4/5. In general with S stages and M
        micro-batches, utilisation is M / (M + S - 1) — the classic
        GPipe bubble formula. That is why the paper says the activations were
        communicated "as soon as they were computed" rather than at the end.

        Why this works in PyTorch with no explicit stream code: CUDA kernel
        launches are ASYNCHRONOUS. The CPU issues stage-0 work for micro-batch
        2 into cuda:0's queue while cuda:1 is still chewing on micro-batch 1.
        The only thing that would break it is a `.item()`, `.cpu()` or an
        explicit `torch.cuda.synchronize()` in the loop — each of those stalls
        the CPU until the GPU catches up and collapses the pipeline. This is
        the single most common way people accidentally serialise a
        model-parallel model.
        """
        B = src.size(1)
        chunks = min(chunks, B)
        sizes = [B // chunks + (1 if i < B % chunks else 0) for i in range(chunks)]
        total, n_tok = 0.0, 0
        losses = []
        start = 0
        for sz in sizes:
            sl = slice(start, start + sz)
            start += sz
            mb_loss = self.loss(src[:, sl], tgt_in[:, sl], tgt_out[:, sl])
            # Re-weight by token count so the micro-batched mean equals the
            # whole-batch mean exactly. Averaging the per-micro-batch means
            # instead would silently up-weight short micro-batches.
            ntok = (tgt_out[:, sl] != PAD).sum().to(self.devices[0])
            losses.append(mb_loss * ntok)
            n_tok = n_tok + ntok
        return torch.stack(losses).sum() / n_tok

    def param_placement(self) -> dict:
        """Where every parameter actually ended up, in bytes per device."""
        by_dev = {}
        for name, p in self.named_parameters():
            d = str(p.device)
            by_dev.setdefault(d, {"params": 0, "bytes": 0, "tensors": []})
            by_dev[d]["params"] += p.numel()
            by_dev[d]["bytes"] += p.numel() * p.element_size()
            by_dev[d]["tensors"].append(name)
        return by_dev


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def timed_steps(step_fn, n_warmup: int = 3, n_iters: int = 10) -> float:
    """Median seconds per step, with warmup and a real device barrier.

    THREE THINGS PEOPLE GET WRONG WHEN TIMING GPU CODE:
      1. No warmup. The first call pays for cuDNN algorithm selection, kernel
         JIT and allocator growth — often 10x the steady-state cost.
      2. No synchronize. CUDA is asynchronous: `t1 - t0` around a launch
         measures how fast Python can *queue* work, not how fast the GPU does
         it. You must synchronize EVERY device, not just the current one.
      3. Reporting the mean. One preemption or clock throttle skews it; the
         median is what you want for "how fast is this normally".
    """
    def sync():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    for _ in range(n_warmup):
        step_fn()
    sync()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        step_fn()
        sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]
