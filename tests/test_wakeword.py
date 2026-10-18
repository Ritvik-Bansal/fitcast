"""Wake word matching and the listener state machine on a synthesized recording (no microphone, no network)."""
import queue
import shutil
import subprocess
from pathlib import Path

import pytest

from fitcast.wakeword import WakeListener, strip_wake, wake_in


def test_wake_variants():
    assert wake_in("hey fitbot find me a jacket")
    assert wake_in("fit bot, what should i wear")
    assert wake_in("fat fitbit find me a casual outfit")
    assert not wake_in("find me a casual outfit")
    assert strip_wake("fat fitbit find me a casual outfit") == "find me a casual outfit"


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("say") is None
                    or not Path("assets/models/vosk-model-small-en-us-0.15").exists(), reason="needs ffmpeg, say and the vosk model")
def test_listener_records_request_after_wake(tmp_path):
    pytest.importorskip("vosk")
    aiff, wav = tmp_path / "q.aiff", tmp_path / "q.wav"
    subprocess.run(["say", "-o", str(aiff), "Fitbot, find me a casual outfit with a hoodie and sweatpants"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    events: queue.Queue = queue.Queue()
    cfg = {"request_file": str(tmp_path / "req.wav"), "silence_s": 0.6}
    lst = WakeListener(cfg, on_wake=lambda: events.put(("wake", None)), on_request=lambda p, h: events.put(("request", (p, h))),
                       on_error=lambda m: events.put(("error", m)), source=str(wav))
    lst.start()
    lst.join(timeout=30)
    kinds = []
    while not events.empty():
        kinds.append(events.get_nowait())
    names = [k for k, _ in kinds]
    assert "wake" in names and "request" in names, kinds
    path, hint = next(v for k, v in kinds if k == "request")
    assert Path(path).stat().st_size > 16000
    assert "hoodie" in hint or "casual" in hint
