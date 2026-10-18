"""Hands-free trigger for the stylist: an offline recogniser (Vosk, small English model) listens to the laptop
microphone all the time; when the wake phrase ("Magic mirror") is heard the following utterance is recorded until a pause
and handed over as a WAV (plus the rough local transcript) for the stylist to answer.

Audio comes from ffmpeg (avfoundation microphone, or a file for tests) as 16 kHz mono PCM on a pipe. Nothing here
touches the network.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("fitcast.wakeword")

_DEFAULT_VARIANTS = ("magic mirror", "magic mirrors", "magic mira", "magic mere", "magic mirra", "match mirror", "magic near", "magic marer", "magic miro")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z ]+", " ", text.lower()).strip()


_WAKE_RE = re.compile(r"\b(magic|magik|match|majik)\s*(mirror|mirrors|mira|mere|mirra|miro|near|marer|mere|mara)\b")


def wake_in(text: str, variants: tuple[str, ...] | list[str] = _DEFAULT_VARIANTS) -> bool:
    """True when any wake variant occurs in `text` (spaces ignored, so 'fit bot' == 'fitbot'), or the small model's
    usual mishearings of "Magic mirror"."""
    t = normalise(text)
    squashed = t.replace(" ", "")
    if any(v.replace(" ", "") in squashed for v in variants):
        return True
    return _WAKE_RE.search(t) is not None


def strip_wake(text: str, variants: tuple[str, ...] | list[str] = _DEFAULT_VARIANTS) -> str:
    """The request part of a transcript: everything after the last wake variant."""
    t = normalise(text)
    best = -1
    for v in variants:
        i = t.rfind(v)
        if i >= 0:
            best = max(best, i + len(v))
    return t[best:].strip() if best >= 0 else t


_REQUEST_WORDS = {
    "outfit", "outfits", "wear", "wearing", "dress", "look", "style", "fit", "fits", "clothes", "clothing",
    "shirt", "tee", "t", "top", "tops", "hoodie", "hoody", "sweater", "sweatshirt", "crewneck", "jacket", "flannel",
    "polo", "jersey", "fleece", "overshirt", "jeans", "pants", "trousers", "bottoms", "cargo", "cargos", "joggers",
    "sweatpants", "chinos", "khakis", "track", "shorts", "denim", "baggy", "flared", "flare", "skinny", "wide",
    "red", "blue", "black", "white", "green", "grey", "gray", "navy", "pink", "yellow", "orange", "purple", "brown",
    "beige", "khaki", "light", "dark", "interview", "casual", "formal", "professional", "party", "gym", "school",
    "stussy", "essentials", "nike", "carhartt", "lakers", "chrome", "hearts", "levis", "champion", "dickies",
}


def request_like(text: str, keywords: set[str] = _REQUEST_WORDS, min_hits: int = 2) -> bool:
    """True when a transcript reads like a clothing request: at least `min_hits` clothing / colour / brand words."""
    words = normalise(text).split()
    hits = sum(1 for w in words if w in keywords)
    return len(words) >= 3 and hits >= min_hits


class WakeListener(threading.Thread):
    """Background thread. Callbacks run on this thread: hand the data to the main loop through a queue.

    on_wake():                     the wake word was heard, recording the request now
    on_request(wav_path, hint):    request audio written (16 kHz mono WAV) + rough transcript ("" when unknown)
    on_error(message):             microphone / model problem (listener stops)
    """

    def __init__(self, cfg: dict, on_wake: Callable[[], None], on_request: Callable[[Path, str], None],
                 on_error: Callable[[str], None] | None = None, source: str | None = None,
                 extra_keywords: set[str] | None = None) -> None:
        super().__init__(name="wakeword", daemon=True)
        self.cfg = cfg
        self.auto_request = bool(cfg.get("auto_request", True))
        self.keywords = set(_REQUEST_WORDS) | set(extra_keywords or ())
        self.min_hits = int(cfg.get("request_min_hits", 2))
        self._hist: deque[bytes] = deque(maxlen=int(float(cfg.get("history_s", 12)) / 0.1))
        self._last_final = 0.0
        self.on_wake, self.on_request, self.on_error = on_wake, on_request, on_error or (lambda m: None)
        self.source = source                                   # None = microphone; path = decode a file (tests)
        self.variants = tuple(cfg.get("wake_words", list(_DEFAULT_VARIANTS)))
        self.rate = 16000
        self.chunk = int(self.rate * 0.1) * 2                  # 100 ms of int16
        self.silence_rms = float(cfg.get("silence_rms", 250))
        self.silence_s = float(cfg.get("silence_s", 0.9))
        self.ack_mute_s = float(cfg.get("ack_mute_s", 0.8))
        self.min_speech_s = float(cfg.get("min_speech_s", 0.5))
        self.max_request_s = float(cfg.get("max_request_s", 10))
        self.pre_roll = int(float(cfg.get("pre_roll_s", 0.6)) / 0.1)
        self.out_path = Path(cfg.get("request_file", "out/wake_request.wav"))
        self._quit = threading.Event()
        self._proc: subprocess.Popen | None = None
        self.state = "idle"
        self.mute_until = 0.0                                  # the booth's own voice must not wake it

    def stop(self) -> None:
        self._quit.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()

    # -- audio source ------------------------------------------------------------------------------
    def _device(self) -> str:
        """avfoundation audio spec: `device_name` (substring, e.g. 'MacBook') resolved against the current device
        list, else the configured index spec (':0'). Devices re-number when a phone or headset appears."""
        name = str(self.cfg.get("device_name", "")).strip()
        if name:
            try:
                r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                                   capture_output=True, text=True, timeout=10)
                audio = False
                for line in (r.stderr + r.stdout).splitlines():
                    if "audio devices" in line:
                        audio = True
                        continue
                    m = re.search(r"\] \[(\d+)\] (.+)$", line)
                    if audio and m and name.lower() in m.group(2).lower():
                        return f":{m.group(1)}"
            except (OSError, subprocess.TimeoutExpired):
                pass
            log.warning("audio device %r not found; using %s", name, self.cfg.get("device", ":0"))
        return str(self.cfg.get("device", ":0"))

    def _open(self) -> subprocess.Popen:
        if self.source is None:
            src = ["-f", "avfoundation", "-i", self._device()]
        else:
            src = ["-re", "-i", str(self.source)]
        cmd = ["ffmpeg", "-loglevel", "error"] + src + ["-ac", "1", "-ar", str(self.rate), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)

    def run(self) -> None:  # noqa: C901 - one state machine, kept in one place on purpose
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError:
            self.on_error("vosk not installed: uv pip install --python .venv/bin/python vosk")
            return
        model_dir = Path(self.cfg.get("model_dir", "assets/models/vosk-model-small-en-us-0.15"))
        if not model_dir.exists():
            self.on_error(f"wake-word model missing at {model_dir}: download vosk-model-small-en-us-0.15 into assets/models/")
            return
        SetLogLevel(-1)
        rec = KaldiRecognizer(Model(str(model_dir)), self.rate)
        try:
            self._proc = self._open()
        except FileNotFoundError:
            self.on_error("ffmpeg not found (brew install ffmpeg)")
            return
        ring: deque[bytes] = deque(maxlen=self.pre_roll)
        buf: list[bytes] = []
        silence, heard, t_start, speech_s = 0.0, False, 0.0, 0.0
        log.info("wake word listening (%s)", "microphone" if self.source is None else self.source)
        restarts, last_beat, beat_rms, beat_n = 0, time.perf_counter(), 0.0, 0
        gain = float(self.cfg.get("gain", 1.0))
        while not self._quit.is_set():
            data = self._proc.stdout.read(self.chunk)
            if not data:
                if self.source is not None or restarts >= int(self.cfg.get("max_restarts", 20)):
                    break
                err = (self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else "").strip()
                log.warning("microphone stream ended (%s); reopening", err[:120] or "no error text")
                time.sleep(1.0)
                restarts += 1
                try:
                    self._proc = self._open()
                except FileNotFoundError:
                    break
                continue
            if gain != 1.0:
                a = np.frombuffer(data, np.int16).astype(np.float32) * gain
                data = np.clip(a, -32768, 32767).astype(np.int16).tobytes()
            # heartbeat: microphone level every 5 s so a dead or wrong microphone is visible in the log
            level = float(np.sqrt(np.mean(np.frombuffer(data, np.int16).astype(np.float32) ** 2)))
            beat_rms += level; beat_n += 1
            if time.perf_counter() - last_beat > 5.0:
                log.info("mic level rms %.0f (%s)", beat_rms / max(beat_n, 1), self.state)
                last_beat, beat_rms, beat_n = time.perf_counter(), 0.0, 0
            ring.append(data)
            self._hist.append(data)
            if self.state == "idle":
                if time.perf_counter() < self.mute_until:
                    rec.Reset()
                    continue
                final = rec.AcceptWaveform(data)
                if final:
                    text = json.loads(rec.Result()).get("text", "")
                else:
                    text = json.loads(rec.PartialResult()).get("partial", "")
                if final and text and self.auto_request and not wake_in(text, self.variants) \
                        and request_like(text, self.keywords, self.min_hits):
                    # no wake word needed: the sentence itself is a clothing request -> send that utterance
                    n = min(len(self._hist), int((time.perf_counter() - self._last_final) / 0.1) + 8)
                    log.info("request heard without wake word: %r (%.1f s of audio)", text, n * 0.1)
                    self._emit(list(self._hist)[-n:], rec, hint=text)
                    self._last_final = time.perf_counter()
                    continue
                if final:
                    self._last_final = time.perf_counter()
                if text and (time.perf_counter() - getattr(self, "_last_text_log", 0.0) > 3.0 or "magic" in text or "mirror" in text):
                    log.info("heard: %r", text[-60:])
                    self._last_text_log = time.perf_counter()
                if text and wake_in(text, self.variants):
                    log.info("wake word heard: %r", text)
                    self.state = "listening"
                    buf, silence, heard, t_start = list(ring), 0.0, False, time.perf_counter()
                    speech_s = 0.0
                    rec.Reset()
                    self.on_wake()
                continue
            # listening: collect until a pause after speech, or the maximum length. The first `ack_mute_s` after
            # the wake word are the booth's own chime / voice and never count as speech; neither does anything
            # while the booth is speaking (`mute_until`).
            if time.perf_counter() < self.mute_until:
                t_start = max(t_start, time.perf_counter() - self.ack_mute_s)   # the clock restarts after the speech
                continue
            buf.append(data)
            rec.AcceptWaveform(data)
            rms = float(np.sqrt(np.mean(np.frombuffer(data, np.int16).astype(np.float32) ** 2)))
            since_wake = time.perf_counter() - t_start
            if since_wake < self.ack_mute_s:
                continue
            if rms >= self.silence_rms:
                speech_s += 0.1
                silence = 0.0
                if speech_s >= self.min_speech_s:
                    heard = True
            else:
                silence += 0.1
            if heard and silence >= self.silence_s:
                self._emit(buf, rec)
                buf, silence, heard, speech_s = [], 0.0, False, 0.0
            elif time.perf_counter() - t_start > self.max_request_s:
                if heard:
                    self._emit(buf, rec)
                else:                                              # wake word, then nothing: no request
                    log.info("no request after the wake word; back to idle")
                    rec.Reset()
                    self.state = "idle"
                    self.on_request(None, "")
                buf, silence, heard, speech_s = [], 0.0, False, 0.0
        if self.state == "listening" and buf and heard:
            self._emit(buf, rec)                                   # file source ended mid-request
        err = (self._proc.stderr.read().decode(errors="replace") if self._proc and self._proc.stderr else "").strip()
        if err and not self._quit.is_set():
            self.on_error(f"ffmpeg: {err[:160]}")
        self.state = "stopped"

    def _emit(self, buf: list[bytes], rec, hint: str | None = None) -> None:
        if hint is None:
            hint = strip_wake(json.loads(rec.FinalResult()).get("text", ""), self.variants)
            rec.Reset()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes(b"".join(buf))
        self.state = "idle"
        log.info("request recorded: %.1f s, local transcript %r", len(buf) * 0.1, hint)
        self.on_request(self.out_path, hint)
