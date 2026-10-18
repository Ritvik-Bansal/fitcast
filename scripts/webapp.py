#!/usr/bin/env python
"""Tiny local web UI for the booth (stdlib only):

    .venv/bin/python scripts/webapp.py            # http://localhost:8765

  /        paste a product URL -> runs scripts/garment_from_url.py -> the outfit appears in the running demo
  /cart    the cart filled by the raise-both-hands gesture (out/cart.json), a plain list of links
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
CART = ROOT / "out" / "cart.json"
RELOAD = ROOT / "out" / "reload.now"
PY = ROOT / ".venv" / "bin" / "python"
_lock = threading.Lock()


def _run_fetch(url: str, part: str, name: str) -> dict:
    cmd = [str(PY), str(ROOT / "scripts" / "garment_from_url.py"), url]
    if part in ("top", "bottom"):
        cmd += ["--part", part]
    if name:
        cmd += ["--name", name[:28]]
    with _lock:                                   # one build at a time
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=240)
    out = (r.stdout + "\n" + r.stderr).strip()
    ok = r.returncode == 0
    if ok:
        RELOAD.touch()                            # the running demo reloads its outfits
    tail = "\n".join(out.splitlines()[-12:])
    return {"ok": ok, "code": r.returncode, "log": tail}


def _excluded(folder: str) -> bool:
    """Folders the demo hides (config garments.exclude_prefixes)."""
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        return folder.startswith(tuple(cfg.get("garments", {}).get("exclude_prefixes") or []))
    except Exception:  # noqa: BLE001
        return False


class H(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8", code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send((WEB / "index.html").read_bytes())
        elif p == "/cart":
            self._send((WEB / "cart.html").read_bytes())
        elif p == "/api/cart":
            data = json.loads(CART.read_text()) if CART.exists() else []
            self._send(json.dumps(data).encode(), "application/json")
        elif p == "/api/outfits":
            items = []
            for d in sorted((ROOT / "assets" / "garments").glob("[0-9]*/meta.yaml")):
                if _excluded(d.parent.name):
                    continue
                name = ""
                for line in d.read_text().splitlines():
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip("'\"")
                        break
                items.append({"folder": d.parent.name, "name": name,
                              "top": name.split(" + ")[0], "bottom": name.split(" + ")[-1]})
            self._send(json.dumps(items).encode(), "application/json")
        elif p.startswith("/thumb/") or p.startswith("/part/"):
            parts = p.strip("/").split("/")
            folder = (ROOT / "assets" / "garments" / parts[1]).resolve()
            if not str(folder).startswith(str((ROOT / "assets" / "garments").resolve())) or not folder.is_dir():
                self._send(b"not found", "text/plain", 404)
                return
            fname = "thumb.png" if parts[0] == "thumb" else ("top_front.png" if parts[2] == "top" else "bottom_front.png")
            f = folder / fname
            if f.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "max-age=60")
                self.send_header("Content-Length", str(f.stat().st_size))
                self.end_headers()
                self.wfile.write(f.read_bytes())
            else:
                self._send(b"not found", "text/plain", 404)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:  # noqa: N802
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n).decode()
        if p == "/api/add":
            q = parse_qs(raw)
            url = (q.get("url") or [""])[0].strip()
            if not url:
                self._send(json.dumps({"ok": False, "log": "no url"}).encode(), "application/json", 400)
                return
            res = _run_fetch(url, (q.get("part") or ["auto"])[0], (q.get("name") or [""])[0])
            self._send(json.dumps(res).encode(), "application/json")
        elif p == "/api/cart/clear":
            CART.write_text("[]")
            self._send(b"{}", "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, fmt, *args):  # quiet
        sys.stderr.write("webapp: " + fmt % args + "\n")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"FitCast web UI: http://localhost:{port}  (cart at /cart)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
