"""Generate ASCII-safe copies of prompts/*.md for LaTeX \\VerbatimInput.

pdflatex with [utf8]inputenc cannot ingest some unicode glyphs that appear
in the source prompts. We mirror prompts/ into generated_prompts/ with the
following substitutions, which keep the meaning of each glyph in a fixed-
width display:

    em-dash    U+2014 ->  --
    arrow      U+2192 ->  ->
    leq        U+2264 ->  <=
    geq        U+2265 ->  >=
    lesssim    U+2272 ->  <~
"""
from __future__ import annotations

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_prompts")

SUBS = {
    "—": "--",
    "→": "->",
    "≤": "<=",
    "≥": ">=",
    "≲": "<~",
}


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".md"):
            continue
        text = open(os.path.join(SRC, name)).read()
        for k, v in SUBS.items():
            text = text.replace(k, v)
        out = os.path.join(DST, name)
        with open(out, "w") as f:
            f.write(text)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
