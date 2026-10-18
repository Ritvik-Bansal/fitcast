"""Gemini-backed outfit picker ("stylist"): a natural-language request, typed or spoken, becomes one top and one
bottom from the loaded catalogue, a one-sentence spoken answer and a few shopping suggestions.

Optional feature: the demo is fully offline without it. Requests run in a background thread; `poll()` hands the
reply to the main loop. No third-party dependencies (urllib, ffmpeg for the microphone, macOS `say` for speech).
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .types import Outfit

log = logging.getLogger("fitcast.stylist")


def load_shop_links(path: str | Path) -> dict[str, list[dict]]:
    """Curated 'where to buy' entries per garment name (`assets/shop_links.yaml`); {} when the file is missing."""
    import yaml
    p = Path(path)
    if not p.exists():
        log.warning("shop links file %s missing: shopping suggestions come from the model only", p)
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    out: dict[str, list[dict]] = {}
    for name, entries in raw.items():
        out[str(name)] = [{"store": str(e.get("store", "")), "item": str(e.get("item", "")),
                           "price_usd": e.get("price_usd"), "url": str(e.get("url", ""))} for e in (entries or [])]
    return out

_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class StylistReply:
    """Result of one request. `top_index` / `bottom_index` index the catalogue list given to the stylist."""
    question: str
    top_index: int | None
    bottom_index: int | None
    say: str
    shopping: list[dict] = field(default_factory=list)
    model: str = ""
    error: str = ""
    latency_s: float = 0.0


def load_api_key(cfg: dict) -> str | None:
    """Key from the environment (GEMINI_API_KEY) or `cfg['key_file']` (gitignored). None when neither exists."""
    import os
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    p = Path(cfg.get("key_file", "secrets/gemini.key"))
    if p.exists():
        k = p.read_text().strip()
        if k:
            return k
    log.error("Gemini key missing: set GEMINI_API_KEY or write it to %s (stylist disabled)", p)
    return None


class GeminiStylist:
    """Catalogue-aware outfit picker. `outfits` is the gallery list; replies index into it."""

    def __init__(self, cfg: dict, outfits: list[Outfit], api_key: str) -> None:
        self.cfg = cfg
        self.outfits = outfits
        self.key = api_key
        self.models = [str(m) for m in cfg.get("models", ["gemini-3.8-flash"])]
        self.timeout = float(cfg.get("timeout_s", 40))
        self.temperature = float(cfg.get("temperature", 0.4))
        self.grounding = bool(cfg.get("grounding", False))
        self._q: queue.Queue = queue.Queue()
        self._busy = threading.Event()
        self.links = load_shop_links(cfg.get("links_file", "assets/shop_links.yaml"))
        # tops: every outfit with a top; bottoms: first outfit per distinct bottom name (the demo pairs repeat "Jeans")
        hide_t, hide_b = set(cfg.get("hide_tops") or []), set(cfg.get("hide_bottoms") or [])
        seen_t: set[str] = set()
        self.top_ids = []
        for i, o in enumerate(outfits):
            name = _top_name(o)
            if o.top_front is not None and name not in seen_t and name not in hide_t:
                seen_t.add(name)
                self.top_ids.append(i)
        seen: set[str] = set()
        self.bottom_ids = []
        for i, o in enumerate(outfits):
            name = _bottom_name(o)
            if o.bottom_front is not None and name not in seen and name not in hide_b:
                seen.add(name)
                self.bottom_ids.append(i)

    # -- catalogue ----------------------------------------------------------------------------
    def catalog(self) -> str:
        """Compact catalogue text with stable ids (T<n> tops, B<n> bottoms) for the prompt."""
        tops = [f"T{k}: {_top_name(self.outfits[i])} ({self.outfits[i].sleeve} sleeve)" for k, i in enumerate(self.top_ids)]
        bots = [f"B{k}: {_bottom_name(self.outfits[i])}" for k, i in enumerate(self.bottom_ids)]
        return "TOPS\n" + "\n".join(tops) + "\nBOTTOMS\n" + "\n".join(bots)

    def known_links(self, idx: int | None, part: str) -> list[dict]:
        """Curated links for the top (part='top') or bottoms (part='bottom') of outfit `idx`."""
        if idx is None:
            return []
        name = _top_name(self.outfits[idx]) if part == "top" else _bottom_name(self.outfits[idx])
        return list(self.links.get(name, []))

    def _links_text(self) -> str:
        lines = []
        for k, i in enumerate(self.top_ids):
            for e in self.links.get(_top_name(self.outfits[i]), [])[:1]:
                lines.append(f"T{k}: {e['item']} at {e['store']} ~${e['price_usd']} {e['url']}")
        for k, i in enumerate(self.bottom_ids):
            for e in self.links.get(_bottom_name(self.outfits[i]), [])[:1]:
                lines.append(f"B{k}: {e['item']} at {e['store']} ~${e['price_usd']} {e['url']}")
        return "\n".join(lines)

    def _prompt(self, question: str | None, hint: str | None = None) -> str:
        if question:
            req = f'User request: "{question}"'
        else:
            req = ("The user's request is spoken in the attached audio; it may start with the wake word 'Fitbot' "
                   "(ignore that word).")
            if hint:
                req += f' A rough offline transcript (may contain errors): "{hint}"'
        links = self._links_text()
        link_rule = ("\n\nKNOWN PURCHASE LINKS (verified, use them for the chosen pieces):\n" + links) if links else ""
        return (
            "You are the stylist of a projection-mapped virtual try-on booth. The booth can only show the garments in "
            "this catalogue, one top and one bottom at a time:\n\n" + self.catalog() + link_rule + "\n\n" + req + "\n\n"
            "If the audio or text contains no actual clothing request (only the wake word, noise, silence or an unrelated "
            "remark), set top_id and bottom_id to null and say so in `say`; never invent an outfit for an empty request. "
            "Otherwise pick exactly one top id and one bottom id that best fit the request (occasion, formality, colours, "
            "budget); a named brand or colour in the request must be matched when the catalogue has it. "
            "If the request names garments the catalogue does not have, pick the closest match and say so briefly. "
            "For shopping, list the known links of the chosen top and bottom; add at most one more real store suggestion "
            "only if it helps the budget. If a budget is given keep the total under it. Reply with JSON only, no markdown, "
            'with keys: "heard" (the request in one line), "top_id", "bottom_id", "say" (one spoken sentence, at most '
            '25 words, naming the outfit and why it fits), "shopping" (list of {"item","store","price_usd","url"}).'
        )

    # -- requests (background) ---------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def ask_text(self, question: str) -> bool:
        """Start a request for a typed question. False when another request is still running."""
        return self._start([{"text": self._prompt(question)}], question)

    def ask_audio(self, wav_path: str | Path, hint: str | None = None) -> bool:
        """Start a request for a recorded question (16 kHz mono WAV). The model transcribes it itself; `hint` is an
        optional rough local transcript."""
        data = base64.b64encode(Path(wav_path).read_bytes()).decode()
        parts = [{"text": self._prompt(None, hint)}, {"inline_data": {"mime_type": "audio/wav", "data": data}}]
        return self._start(parts, hint or "(voice)")

    def _start(self, parts: list[dict], question: str) -> bool:
        if self._busy.is_set():
            log.warning("stylist busy; request dropped")
            return False
        self._busy.set()
        threading.Thread(target=self._worker, args=(parts, question), daemon=True, name="stylist").start()
        return True

    def poll(self) -> StylistReply | None:
        """Non-blocking: the next finished reply, or None."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def _worker(self, parts: list[dict], question: str) -> None:
        t0 = time.perf_counter()
        try:
            text, model = self._call(parts)
            reply = self.parse(text, question)
            reply.model = model
        except Exception as e:  # noqa: BLE001 - surfaced to the operator, never crashes the demo
            reply = StylistReply(question=question, top_index=None, bottom_index=None, say="", error=str(e)[:200])
        reply.latency_s = time.perf_counter() - t0
        self._q.put(reply)
        self._busy.clear()

    def _call(self, parts: list[dict]) -> tuple[str, str]:
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": self.temperature}}
        if self.grounding:
            body["tools"] = [{"google_search": {}}]
        else:
            body["generationConfig"]["responseMimeType"] = "application/json"
        last = ""
        for model in self.models:
            code, payload = _post(_API.format(model=model), body, self.key, self.timeout)
            if code == 200:
                cands = payload.get("candidates") or []
                if not cands:
                    last = f"{model}: empty reply ({payload.get('promptFeedback')})"
                    continue
                text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
                return text, model
            last = f"{model}: HTTP {code} {str(payload)[:120]}"
            log.warning("stylist %s", last)
            if code not in (404, 429, 500, 503):
                break
        raise RuntimeError(last or "no model answered")

    # -- parsing --------------------------------------------------------------------------------
    def parse(self, text: str, question: str) -> StylistReply:
        """JSON (possibly fenced) -> StylistReply; ids outside the catalogue become None."""
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON in reply: {text[:120]!r}")
        d = json.loads(m.group(0))
        top = _id_index(d.get("top_id"), "T", self.top_ids)
        bot = _id_index(d.get("bottom_id"), "B", self.bottom_ids)
        shopping = []
        for s in d.get("shopping") or []:
            if isinstance(s, dict):
                shopping.append({"item": str(s.get("item", ""))[:60], "store": str(s.get("store", ""))[:40],
                                 "price_usd": s.get("price_usd"), "url": str(s.get("url", ""))[:120]})
        heard = str(d.get("heard") or question)
        # curated links for the chosen pieces always come first (verified URLs); the model's extras follow, deduplicated
        merged: list[dict] = []
        for e in self.known_links(top, "top")[:1] + self.known_links(bot, "bottom")[:1] + shopping:
            if e.get("url") and all(e["url"] != m["url"] for m in merged):
                merged.append(e)
        return StylistReply(question=heard, top_index=top, bottom_index=bot, say=str(d.get("say", ""))[:220],
                            shopping=merged[:4])


def _id_index(value, prefix: str, ids: list[int]) -> int | None:
    if value is None:
        return None
    m = re.search(r"(\d+)", str(value))
    if not m:
        return None
    k = int(m.group(1))
    return ids[k] if 0 <= k < len(ids) else None


def _top_name(o: Outfit) -> str:
    return o.name.split(" + ")[0]


def _bottom_name(o: Outfit) -> str:
    return o.name.split(" + ")[-1]


def _post(url: str, body: dict, key: str, timeout: float) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw[:200]


# -- microphone + speech (macOS, no dependencies) -----------------------------------------------
class Recorder:
    """Push-to-talk recorder via ffmpeg's avfoundation input: `start()`, then `stop()` returns the WAV path."""

    def __init__(self, cfg: dict) -> None:
        self.device = str(cfg.get("record_device", ":0"))
        self.path = Path(cfg.get("record_file", "out/ask.wav"))
        self.max_s = float(cfg.get("record_max_s", 12))
        self.proc: subprocess.Popen | None = None
        self.started = 0.0

    @property
    def recording(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        if self.recording:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-i", self.device,
               "-t", str(self.max_s), "-ac", "1", "-ar", "16000", str(self.path)]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            log.error("ffmpeg not found: install it (brew install ffmpeg) or type questions into out/ask.txt")
            return False
        self.started = time.perf_counter()
        return True

    def stop(self) -> Path | None:
        """Stop and return the file when it holds at least ~0.5 s of audio."""
        if self.proc is None:
            return None
        p, self.proc = self.proc, None
        try:
            if p.poll() is None:
                p.stdin.write(b"q")
                p.stdin.flush()
                p.wait(timeout=4)
        except Exception:  # noqa: BLE001
            p.kill()
        err = (p.stderr.read().decode(errors="replace") if p.stderr else "").strip()
        if err:
            log.warning("ffmpeg: %s", err[:200])
        if self.path.exists() and self.path.stat().st_size > 16000:
            return self.path
        log.warning("recording too short or microphone blocked (grant the terminal microphone access)")
        return None


def speak(text: str, rate: int = 185) -> None:
    """Read `text` aloud with macOS `say` without blocking (no-op when unavailable)."""
    if not text:
        return
    try:
        subprocess.Popen(["say", "-r", str(rate), text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        log.info("say not available; stylist reply: %s", text)


# -- sharing the links --------------------------------------------------------------------------
def make_qr(url: str, size_px: int) -> "np.ndarray | None":
    """QR code image (size_px square, BGR) for `url`, or None when OpenCV lacks the encoder."""
    import cv2
    import numpy as np
    try:
        enc = cv2.QRCodeEncoder_create()
        qr = enc.encode(url)
    except Exception as e:  # noqa: BLE001
        log.warning("QR encoder unavailable: %s", e)
        return None
    qr = cv2.resize(qr, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


def links_text(reply: "StylistReply", outfit_name: str) -> str:
    lines = [f"FitCast: {outfit_name}"]
    for e in reply.shopping:
        price = f" ~${e['price_usd']}" if e.get("price_usd") not in (None, "") else ""
        lines.append(f"- {e['item']} ({e['store']}{price}): {e['url']}")
    return "\n".join(lines)


def share_links(reply: "StylistReply", outfit_name: str, cfg: dict) -> str:
    """Write the links to `out/last_links.txt` and, when `imessage_to` is set, send them with Messages.app."""
    text = links_text(reply, outfit_name)
    Path("out").mkdir(exist_ok=True)
    Path(cfg.get("links_out", "out/last_links.txt")).write_text(text + "\n")
    to = str(cfg.get("imessage_to", "")).strip()
    if to:
        script = ('tell application "Messages"\n'
                  '  set s to 1st account whose service type = iMessage\n'
                  '  send "%s" to participant "%s" of s\n'
                  'end tell') % (text.replace("\\", "\\\\").replace('"', '\\"'), to)
        try:
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("links sent by iMessage to %s", to)
        except FileNotFoundError:
            log.warning("osascript missing; links only in out/last_links.txt")
    return text
