"""Assemble notebooks/seq2seq_from_scratch.ipynb from the parts in nbsrc/.

The notebook is GENERATED, not hand-edited. The source of truth is the
`part*.txt` files here, which are plain text split into cells by delimiter
lines:

    #%%md      -> everything until the next delimiter is a markdown cell
    #%%py      -> ... is a Python code cell

Why not edit the .ipynb directly? Because .ipynb is JSON with every line
escaped, which makes diffs unreadable and merges impossible. Keeping the prose
and code as plain text means `git diff` shows you the sentence that changed.

Usage:
    python nbsrc/build_notebook.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "notebooks", "seq2seq_from_scratch.ipynb")


def parse(path):
    cells = []
    kind, buf = None, []

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if not src.strip():
            return
        # nbformat stores source as a list of lines, each keeping its newline
        # except the last. Splitting this way keeps GitHub's renderer happy.
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": source,
                          "execution_count": None, "outputs": []})

    with open(path, encoding="utf-8") as f:
        for line in f.read().split("\n"):
            if line.rstrip() in ("#%%md", "#%%py"):
                flush()
                kind = line.rstrip()[3:]
                buf = []
            else:
                buf.append(line)
    flush()
    return cells


def main():
    parts = sorted(glob.glob(os.path.join(HERE, "part*.txt")))
    if not parts:
        raise SystemExit("no part*.txt files found in nbsrc/")
    cells = []
    for p in parts:
        n = parse(p)
        print(f"  {os.path.basename(p):16s} {len(n):3d} cells")
        cells += n

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"wrote {OUT}: {len(cells)} cells ({md} markdown, {len(cells)-md} code), "
          f"{os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
