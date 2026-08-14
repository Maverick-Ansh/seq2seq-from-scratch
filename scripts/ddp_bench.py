"""Real data parallelism (DistributedDataParallel) on 2 GPUs, benchmarked.

WHY THIS IS A SEPARATE SCRIPT AND NOT A NOTEBOOK CELL
-----------------------------------------------------
`nn.DataParallel` is single-process, multi-thread: one Python process drives
both GPUs. That is convenient in a notebook and it is also why it is slow —
see below. Real data parallelism is *multi-process*: one process per GPU, each
with its own Python interpreter, its own CUDA context and its own copy of the
model, synchronising gradients through NCCL collectives. You cannot start those
processes from inside a running notebook kernel, so this is launched with

    torchrun --nproc_per_node=2 scripts/ddp_bench.py

THE DIFFERENCE, WHICH IS THE WHOLE POINT
-----------------------------------------
nn.DataParallel, EVERY forward pass:
    1. replicate the full model from GPU0 to GPU1     <- 1.5 GB copy, every step
    2. scatter the batch
    3. run both replicas (in threads, fighting the GIL)
    4. gather all outputs back to GPU0                <- another big copy
    5. backward on GPU0 only, then discard the replica
  The replication in step 1 is proportional to PARAMETER COUNT and happens
  every single step. For a 384M-parameter model that is ~1.5 GB over PCIe per
  step — comparable to the entire compute time. This is why DataParallel is
  deprecated and why our notebook measured a 1.00x "speedup".

DistributedDataParallel:
    1. each process already HOLDS its own model. Nothing is replicated.
    2. each process loads its own slice of the batch
    3. forward + backward entirely locally
    4. during backward, gradients are ALL-REDUCED in buckets, overlapping with
       the rest of the backward pass — so most of the communication is hidden
       behind compute that was going to happen anyway.
  Communication is still O(parameters) per step, but it happens once, it is
  overlapped, and NCCL uses a ring algorithm that is bandwidth-optimal.

WHAT TO EXPECT ON 2x T4. DDP should land meaningfully above 1.0x but below
2.0x: the all-reduce of 384M fp32 parameters is 1.5 GB per step over PCIe, and
a T4 pair has no NVLink. Data parallelism shines when the model is small
relative to the compute per example; a 384M-parameter model with only 30
timesteps of work is close to the worst case for it.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seq2seq.model import Seq2Seq, sequence_loss, paper_config


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    # NCCL is the GPU-to-GPU backend; it uses P2P/DMA where available and falls
    # back to host staging where not. (Chapter 0 confirmed P2P is available.)
    dist.init_process_group("nccl")

    cfg = paper_config()
    B_GLOBAL, S, T = 128, 30, 30
    B_LOCAL = B_GLOBAL // world      # each rank handles its own slice

    model = Seq2Seq(**cfg).to(rank)
    # DDP broadcasts rank 0's weights to everyone ONCE, here — not every step.
    ddp = DDP(model, device_ids=[rank])
    opt = torch.optim.SGD(ddp.parameters(), lr=0.7)

    g = torch.Generator().manual_seed(rank)
    src = torch.randint(4, cfg["src_vocab"], (S, B_LOCAL), generator=g).to(rank)
    tin = torch.randint(4, cfg["tgt_vocab"], (T, B_LOCAL), generator=g).to(rank)
    tout = torch.randint(4, cfg["tgt_vocab"], (T, B_LOCAL), generator=g).to(rank)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = sequence_loss(ddp(src, tin), tout)
        loss.backward()          # <- the all-reduce happens inside here
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 5.0)
        opt.step()

    for _ in range(3):           # warmup: cuDNN autotune + NCCL ring setup
        step()
    torch.cuda.synchronize(rank)
    dist.barrier()               # start the clock together

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(rank)
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]

    # Every rank processed B_LOCAL examples in `med` seconds, so global
    # throughput is the GLOBAL batch divided by the (concurrent) step time.
    if rank == 0:
        wps = B_GLOBAL * (S + T) / med
        print(f"DDP_RESULT sec_per_step={med:.4f} words_per_sec={int(wps)} "
              f"world={world} peak_GB={torch.cuda.max_memory_allocated(rank)/1e9:.2f}",
              flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
