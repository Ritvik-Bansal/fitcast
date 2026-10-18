"""App state: which layer gestures edit, what each layer shows, checkout progress.

Both layers (the chosen shirt and the chosen pants) are always drawn; the mode
only decides which one swipes and checkout act on.
"""
from __future__ import annotations

from dataclasses import dataclass

MODES = ("shirt", "pants")
IDLE, CHARGING, PROCESSING, SUCCESS, FAILED = "idle", "charging", "processing", "success", "failed"


@dataclass
class Toast:
    text: str
    until: float


class AppState:
    def __init__(self, cfg, catalog):
        self.cfg = cfg
        self.catalog = catalog
        self.mode = "shirt" if catalog.items("shirt") else "pants"
        self.index = {m: 0 for m in MODES}
        self.checkout = IDLE
        self.checkout_progress = 0.0
        self.checkout_garment = None
        self.checkout_result = None
        self.checkout_until = 0.0
        self.toast: Toast | None = None
        self.debug_overlays = cfg.show_debug_overlays
        self.running = True

    # -- garments -----------------------------------------------------------
    def garment(self, mode: str):
        items = self.catalog.items(mode)
        return items[self.index[mode] % len(items)] if items else None

    @property
    def active(self):
        return self.garment(self.mode)

    def step(self, delta: int, now: float) -> None:
        items = self.catalog.items(self.mode)
        if not items or self.checkout in (PROCESSING, SUCCESS):
            return
        self.index[self.mode] = (self.index[self.mode] + delta) % len(items)
        g = self.active
        self.toast = Toast(f"{g.name}  {g.price_text}", now + 1.2)

    def toggle_mode(self, now: float) -> None:
        other = "pants" if self.mode == "shirt" else "shirt"
        if not self.catalog.items(other):
            self.toast = Toast(f"no {other} in catalog", now + 1.2)
            return
        self.mode = other
        self.toast = Toast(f"{other.upper()} MODE", now + 1.2)

    # -- checkout -------------------------------------------------------------
    @property
    def checkout_ready(self) -> bool:
        return self.checkout in (IDLE, CHARGING) and self.active is not None

    def set_charging(self, progress: float) -> None:
        if self.checkout in (IDLE, CHARGING):
            self.checkout = CHARGING if progress > 0.0 else IDLE
            self.checkout_progress = progress

    def begin_checkout(self, now: float):
        if not self.checkout_ready:
            return None
        self.checkout = PROCESSING
        self.checkout_progress = 1.0
        self.checkout_garment = self.active
        self.checkout_result = None
        return self.checkout_garment

    def finish_checkout(self, result, now: float) -> None:
        if self.checkout != PROCESSING:
            return
        self.checkout_result = result
        self.checkout = SUCCESS if result.ok else FAILED
        self.checkout_until = now + self.cfg.checkout_display_ms / 1000.0

    def tick(self, now: float) -> None:
        if self.checkout in (SUCCESS, FAILED) and now >= self.checkout_until:
            self.checkout = IDLE
            self.checkout_progress = 0.0
        if self.toast and now >= self.toast.until:
            self.toast = None
