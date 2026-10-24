"""TTY presentation with the original banner, low-frequency spinners and streaming."""

from importlib.resources import files
import os
import shutil
import sys
import threading
import time


class TerminalUI:
    def __init__(self, *, stream=None, interactive=None):
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = (self.stream.isatty() if interactive is None else interactive)
        self.color = self.interactive and "NO_COLOR" not in os.environ
        self._stop = threading.Event()
        self._thread = None
        self.response_text = ""
        self.shown_answer = None
        self._open_line = False
        self._reasoning_open = False

    def banner(self):
        if not self.interactive:
            return
        if self.color and shutil.get_terminal_size().columns >= 114:
            art = files("harness").joinpath("banner.ansi").read_text(encoding="utf-8")
            for line in art.splitlines():
                print(line, file=self.stream, flush=True)
                time.sleep(0.015)
        else:
            print("\n  H A R N E S S\n", file=self.stream, flush=True)

    def begin_turn(self):
        self.pause()
        self.response_text = ""
        self.shown_answer = None
        self._reasoning_open = False

    def background(self, message):
        # Plain output stays suitable for piping. The TUI has a separate activity footer.
        pass

    def reasoning(self, text):
        self.pause()
        if not self._reasoning_open:
            self._end_line()
            if not self.color:
                self.stream.write("Reasoning: ")
        self._reasoning_open = True
        self.stream.write(f"\033[90;3m{text}\033[0m" if self.color else text)
        self.stream.flush()
        self._open_line = bool(text) and not text.endswith("\n")

    def message(self, text):
        self.pause()
        self._end_line()
        print(text, file=self.stream, flush=True)

    def read_prompt(self):
        return input("\n> ")

    def memories(self, records, detail=False):
        from .memory_view import render_memories
        width = shutil.get_terminal_size().columns
        self.message(render_memories(records, color=self.color, width=width, detail=detail))

    def approve(self, request, cwd, operation):
        from .tools import terminal_approval
        self.pause()
        return terminal_approval(request, cwd, operation)

    def _spin(self, message):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        index = 0
        while not self._stop.is_set():
            label = f"{frames[index % len(frames)]} {message}"
            if self.color:
                label = f"\033[90m{label}\033[0m"
            self.stream.write(f"\r\033[K{label}")
            self.stream.flush()
            index += 1
            self._stop.wait(0.12)

    def pause(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self.stream.write("\r\033[K")
            self.stream.flush()

    def status(self, event):
        self.pause()
        if event == "response_end":
            self._end_line()
            return
        if event == "generation":
            self.response_text = ""
        messages = {"startup": "Loading local model…", "recall": "Digging through memory…",
                    "generation": "Generating…", "planning": "Preparing plan…",
                    "memory": "Curating memory…"}
        message = messages.get(event, f"Running {event.removeprefix('tool:')}…")
        if self.interactive:
            self._stop.clear()
            self._thread = threading.Thread(target=self._spin, args=(message,), daemon=True)
            self._thread.start()

    def write(self, text):
        self.pause()
        if self._reasoning_open:
            self._end_line()
            self._reasoning_open = False
        self.response_text += text
        self.stream.write(text)
        self.stream.flush()
        self._open_line = bool(text) and not text.endswith("\n")

    def _end_line(self):
        if self._open_line:
            self.stream.write("\n")
            self.stream.flush()
            self._open_line = False

    def answer(self, text):
        self.pause()
        if self.shown_answer == text:
            return
        if self.response_text != text or not text:
            self._end_line()
            print(text, file=self.stream, flush=True)
        else:
            self._end_line()
        self.shown_answer = text

    def close(self):
        self.pause()
        self._end_line()
