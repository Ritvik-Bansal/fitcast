"""Projector window in its own process (pygame), used when cv2 cannot place it.

Started by display.ProjectorWindow as `python -m tryon.projector_child ...`. It
must not import cv2: OpenCV bundles its own SDL2, and two SDL copies in one
process break Cocoa. Frames arrive through a double-buffered shared-memory
block (header: seq, slot), keys go back as lines on stdout, and commands
("fullscreen", "quit") come in on stdin.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from multiprocessing import resource_tracker, shared_memory

HEADER = 64


def _stdin_reader(q: queue.Queue) -> None:
    for line in sys.stdin:
        q.put(line.strip())
    q.put("quit")


def main(argv: list[str]) -> int:
    shm_name, w, h, x, y, disp = argv[0], int(argv[1]), int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5])
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import numpy as np
    import pygame

    shm = shared_memory.SharedMemory(name=shm_name)
    # the parent owns the segment; stop this process's tracker from unlinking it at exit
    resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
    header = np.ndarray((2,), np.uint64, buffer=shm.buf[:16])
    n = w * h * 3

    pygame.display.init()
    disp = min(disp, max(0, len(pygame.display.get_desktop_sizes()) - 1))
    full = True

    def open_window(fullscreen: bool):
        if fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=disp)
        return pygame.display.set_mode((w, h), pygame.NOFRAME, display=disp)

    screen = open_window(full)
    pygame.display.set_caption("Fitcast projector")
    pygame.mouse.set_visible(False)
    names = {pygame.K_LEFT: "left", pygame.K_RIGHT: "right", pygame.K_UP: "up", pygame.K_DOWN: "down",
             pygame.K_SPACE: "space", pygame.K_RETURN: "enter", pygame.K_KP_ENTER: "enter",
             pygame.K_ESCAPE: "esc", pygame.K_BACKSPACE: "backspace"}
    cmds: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(cmds,), daemon=True).start()
    print("ready", flush=True)
    last = -1
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    print("q", flush=True)
                elif ev.type == pygame.KEYDOWN:
                    name = names.get(ev.key) or (ev.unicode.lower() if ev.unicode and ev.unicode.isprintable() else None)
                    if name:
                        print(name, flush=True)
            while not cmds.empty():
                cmd = cmds.get_nowait()
                if cmd == "quit":
                    return 0
                if cmd == "fullscreen":
                    full = not full
                    screen = open_window(full)
            seq = int(header[0])
            if seq != last:
                last = seq
                slot = int(header[1])
                off = HEADER + slot * n
                surf = pygame.image.frombuffer(shm.buf[off:off + n], (w, h), "RGB")
                sw, sh = screen.get_size()
                if (sw, sh) != (w, h):
                    surf = pygame.transform.smoothscale(surf, (sw, sh))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            else:
                time.sleep(0.002)
    finally:
        pygame.quit()
        try:
            shm.close()
        except BufferError:  # a surface still references the buffer; the OS frees it at exit
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
