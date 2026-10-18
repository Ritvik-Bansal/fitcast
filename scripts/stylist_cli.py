#!/usr/bin/env python
"""Ask the Gemini stylist from the command line (no camera / projector needed).

    .venv/bin/python scripts/stylist_cli.py "I have an interview tomorrow, outfit under 80 dollars"
    .venv/bin/python scripts/stylist_cli.py --audio out/ask.wav
    .venv/bin/python scripts/stylist_cli.py --say "casual black top and blue jeans"      # also speaks the answer
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fitcast.garments import load_outfits  # noqa: E402
from fitcast.stylist import GeminiStylist, load_api_key, speak  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="typed request")
    ap.add_argument("--audio", help="WAV file with the spoken request")
    ap.add_argument("--say", action="store_true", help="read the answer aloud (macOS say)")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(a.config))
    outfits = load_outfits(cfg["garments"])
    key = load_api_key(cfg["stylist"])
    if key is None:
        return 2
    st = GeminiStylist(cfg["stylist"], outfits, key)
    if a.audio:
        st.ask_audio(a.audio)
    elif a.question:
        st.ask_text(a.question)
    else:
        ap.error("give a question or --audio")
    while (r := st.poll()) is None:
        time.sleep(0.05)
    if r.error:
        print("ERROR:", r.error)
        return 1
    top = outfits[r.top_index].name.split(" + ")[0] if r.top_index is not None else "(none)"
    bot = outfits[r.bottom_index].name.split(" + ")[-1] if r.bottom_index is not None else "(none)"
    print(f"heard:   {r.question}\nmodel:   {r.model} ({r.latency_s:.1f} s)\noutfit:  {top} + {bot}\nsay:     {r.say}")
    for s in r.shopping:
        print(f"buy:     {s['item']} - {s['store']} ~${s['price_usd']} {s['url']}")
    if a.say:
        speak(r.say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
