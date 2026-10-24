"""Bottom-anchored terminal composer; model and tools run off the UI thread."""

import asyncio
import base64
import queue
import re
import threading
import time

from prompt_toolkit import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, DynamicContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Box, TextArea

from .commands import COMMANDS
from .ui import TerminalUI


# Modified Enter is not a distinct prompt-toolkit key. Decode the CSI-u and
# xterm modifyOtherKeys encodings as LF; CR remains the submit key.
for sequence in ("\x1b[13;2u", "\x1b[27;2;13~"):
    ANSI_SEQUENCES[sequence] = Keys.ControlJ


class TranscriptStream:
    def __init__(self, ui):
        self.ui = ui

    def write(self, text):
        self.ui.dispatch(lambda: self.ui.append(text))
        return len(text)

    def flush(self):
        pass


class ConversationControl(FormattedTextControl):
    def __init__(self, ui):
        self.ui = ui
        self.width = 80
        super().__init__(lambda: ui.wrapped_transcript(self.width), get_cursor_position=ui._cursor)

    def create_content(self, width, height):
        self.width = max(1, width)
        return super().create_content(width, height)

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.ui.scroll_conversation(-3)
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.ui.scroll_conversation(3)
        elif mouse_event.button == MouseButton.LEFT:
            if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
                self.ui.selection_start = mouse_event.position
                self.ui.selection_end = mouse_event.position
                self.ui.selecting = True
                self.ui._scroll_line = self.ui.transcript_window.vertical_scroll
            elif mouse_event.event_type in (MouseEventType.MOUSE_MOVE, MouseEventType.MOUSE_UP):
                if self.ui.selecting:
                    self.ui.selection_end = mouse_event.position
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self.ui.selecting = False
            self.ui.app.invalidate()
        else:
            return NotImplemented
        return None


class ChatUI(TerminalUI):
    def __init__(self, *, input=None, output=None):
        super().__init__(stream=TranscriptStream(self), interactive=True)
        self.transcript = ""
        self.busy = True
        self.approval_pending = False
        self.status_message = ""
        self.background_message = ""
        self.cancelled = threading.Event()
        self._requests = queue.Queue()
        self._approvals = queue.Queue()
        self._loop = None
        self._ui_thread = None
        self._worker = None
        self._scroll_line = None
        self._wrapped_key = None
        self._wrapped = []
        self.selection_start = self.selection_end = None
        self.selecting = False
        self._soft_breaks = set()
        self._visual_line_count = 1
        self.selected_command = 0
        self._suppress_menu = False
        self._transcript_limit = 200_000
        self.composer = TextArea(
            multiline=True, wrap_lines=True, history=InMemoryHistory(),
            read_only=Condition(lambda: self.busy and not self.approval_pending),
            height=Dimension(min=1, max=6), dont_extend_height=True,
            get_line_prefix=lambda line, wrap: [("class:input-marker", "› " if line == 0 and wrap == 0 else "  ")],
            input_processors=[ConditionalProcessor(
                BeforeInput(lambda: [("class:placeholder", self.placeholder)]),
                Condition(lambda: not self.composer.text),
            )],
        )
        self.composer.buffer.on_text_changed += self._text_changed
        self.conversation_control = ConversationControl(self)
        self.transcript_window = Window(self.conversation_control, wrap_lines=False, always_hide_cursor=True)
        self.menu_window = Window(FormattedTextControl(self.menu_fragments),
                                  height=lambda: len(self.matches), style="class:menu")
        self.composer_box = Box(self.composer, padding_top=1, padding_bottom=1,
                                padding_left=1, padding_right=1,
                                style=lambda: "class:approval-input" if self.approval_pending else "class:composer")
        self.chat_root = HSplit([
            self.transcript_window,
            Window(FormattedTextControl(self.status_fragments), height=1),
            ConditionalContainer(self.menu_window, Condition(lambda: self.menu_visible)),
            self.composer_box,
            Window(FormattedTextControl(self.footer_fragments), height=1),
        ])
        from .memory_browser import MemoryBrowser
        from .session_browser import SessionBrowser
        self.memory_browser = MemoryBrowser(self)
        self.session_browser = SessionBrowser(self)
        self._session_views = {}
        self.root = DynamicContainer(lambda: self.memory_browser.container if self.memory_browser.active
                                     else self.session_browser.container if self.session_browser.active else self.chat_root)
        self.app = Application(
            layout=Layout(self.root, focused_element=self.composer),
            full_screen=True, mouse_support=True, key_bindings=self._key_bindings(),
            style=Style.from_dict({
                "": "bg:#101010 #eeeeee", "composer": "bg:#262626 #eeeeee",
                "input-marker": "bold", "placeholder": "#808080",
                "approval-input": "bg:#262626 #00d7ff",
                "approval": "#00d7ff bold", "menu": "bg:#303030 #dddddd",
                "menu-selected": "bg:#484848 #ffffff bold", "muted": "#999999",
                "memory-selected": "bg:#193848 #ffffff bold", "memory-heading": "#00d7ff bold",
            } if self.color else {}),
            input=input, output=output, refresh_interval=0.12, min_redraw_interval=0.04,
        )
        self.app.ttimeoutlen = 0.05
        self.app.timeoutlen = 0.2

    @property
    def placeholder(self):
        if self.approval_pending:
            return "Type yes to approve; Enter denies"
        return "Working…" if self.busy else "Ask Harness to do anything"

    @property
    def matches(self):
        text = self.composer.text
        if not text.startswith("/") or any(char.isspace() for char in text):
            return []
        return [command for command in COMMANDS if command.name.startswith(text)]

    @property
    def menu_visible(self):
        return not self.busy and not self.approval_pending and not self._suppress_menu and bool(self.matches)

    def _text_changed(self, _buffer):
        self.selected_command = 0
        self._suppress_menu = False

    def menu_fragments(self):
        result = []
        for index, command in enumerate(self.matches):
            style = "class:menu-selected" if index == self.selected_command else "class:menu"
            marker = "›" if index == self.selected_command else " "
            line = f" {marker} {command.name:<11} {command.description}"
            width = self.app.output.get_size().columns
            result.append((style, line[:width].ljust(width) + "\n"))
        return result

    def status_fragments(self):
        if not self.status_message:
            return []
        frame = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() / .12) % 10]
        style = "class:approval" if self.approval_pending else "class:muted"
        return [(style, f" {frame} {self.status_message}")]

    def footer_fragments(self):
        if self.approval_pending:
            return [("class:approval", " Only yes approves · Enter / Esc / Ctrl+C deny")]
        if self.menu_visible:
            return [("class:muted", " ↑/↓ choose · Enter/Tab insert · Esc close")]
        if self.selected_text():
            return [("class:muted", " Text selected · Ctrl+C copy · Esc clear · Wheel scroll ")]
        if self.background_message:
            return [("class:muted", f" {self.background_message} · / commands")]
        return [("class:muted", " Enter send · Shift+Enter newline · Wheel/trackpad scroll · Press / for available commands ")]

    def _cursor(self):
        last = self._visual_line_count - 1
        return Point(x=0, y=last if self._scroll_line is None else min(last, self._scroll_line))

    def wrapped_transcript(self, width):
        key = (self.transcript, width)
        if key == self._wrapped_key:
            return self.highlighted_transcript()
        if self._wrapped_key and width != self._wrapped_key[1]:
            self.clear_selection()
        result = []
        column, lines, offset = 0, 1, 0
        self._soft_breaks = set()
        for style, text, *_ in to_formatted_text(ANSI(self.transcript)):
            fragment = ""
            for char in text.expandtabs(4):
                size = get_cwidth(char)
                if char != "\n" and column + size > width:
                    self._soft_breaks.add(offset)
                    fragment += "\n"
                    offset += 1
                    column = 0
                    lines += 1
                fragment += char
                offset += 1
                if char == "\n":
                    column = 0
                    lines += 1
                else:
                    column += size
            result.append((style, fragment))
        self._wrapped_key, self._wrapped, self._visual_line_count = key, result, lines
        return self.highlighted_transcript()

    def selection_range(self):
        if self.selection_start is None or self.selection_end is None:
            return 0, 0
        lines = "".join(text for _, text in self._wrapped).split("\n")

        def index(point):
            row = max(0, min(point.y, len(lines) - 1))
            return sum(len(line) + 1 for line in lines[:row]) + max(0, min(point.x, len(lines[row])))

        return sorted((index(self.selection_start), index(self.selection_end)))

    def selected_text(self):
        if self.selection_start is None or self.selection_end is None:
            return ""
        start, end = self.selection_range()
        text = "".join(text for _, text in self._wrapped)
        return "".join(char for index, char in enumerate(text[start:end], start) if index not in self._soft_breaks)

    def highlighted_transcript(self):
        start, end = self.selection_range()
        if start == end:
            return self._wrapped
        result, offset = [], 0
        for style, text in self._wrapped:
            left, right = max(0, start - offset), min(len(text), end - offset)
            if left < right:
                result.extend([(style, text[:left]), (style + " reverse", text[left:right]), (style, text[right:])])
            else:
                result.append((style, text))
            offset += len(text)
        return result

    def clear_selection(self):
        self.selection_start = self.selection_end = None
        self.selecting = False
        self.app.invalidate()

    def copy_selection(self):
        text = self.selected_text()
        if not text:
            return False
        self.app.clipboard.set_text(text)
        # Explicit Ctrl+C exports the selected text through the terminal's clipboard
        # protocol. The internal clipboard also supports pasting into the composer.
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.app.output.write_raw(f"\x1b]52;c;{encoded}\x07")
        self.app.output.flush()
        self.clear_selection()
        return True

    def scroll_conversation(self, rows):
        info = self.transcript_window.render_info
        height = info.window_height if info else max(1, self.app.output.get_size().rows - 5)
        last_top = max(0, self._visual_line_count - height)
        top = self.transcript_window.vertical_scroll
        target = max(0, min(last_top, top + rows))
        self.transcript_window.vertical_scroll = target
        self._scroll_line = None if rows > 0 and target == last_top else target
        self.app.invalidate()

    def background(self, message):
        def update():
            self.background_message = message
            self.app.invalidate()
        self.dispatch(update)

    def reasoning(self, text):
        self._check_cancelled()
        super().reasoning(text)

    def append(self, text):
        self.transcript += text
        if len(self.transcript) > self._transcript_limit:
            cut = self.transcript.find("\n", len(self.transcript) - self._transcript_limit)
            self.transcript = self.transcript[cut + 1:] if cut >= 0 else self.transcript[-self._transcript_limit:]
            self._scroll_line = None
            self.clear_selection()
        self.app.invalidate()

    def dispatch(self, callback):
        if self._loop is not None and threading.get_ident() != self._ui_thread:
            self._loop.call_soon_threadsafe(callback)
        else:
            callback()

    def _check_cancelled(self):
        if self.cancelled.is_set():
            raise KeyboardInterrupt

    def pause(self):
        self.dispatch(lambda: self._set_status(""))

    def _set_status(self, message):
        self.status_message = message
        self.app.invalidate()

    def status(self, event):
        self._check_cancelled()
        if event == "response_end":
            self._end_line()
            self.pause()
            return
        if event == "generation":
            self.response_text = ""
        labels = {"startup": "Loading local model…", "generation": "Generating…",
                  "recall": "Digging through memory…", "planning": "Preparing plan…",
                  "memory": "Curating memory…"}
        self.dispatch(lambda: self._set_status(labels.get(event, f"Running {event.removeprefix('tool:')}…")))

    def write(self, text):
        self._check_cancelled()
        super().write(text)

    def browse_memories(self, runtime, session, argument=""):
        if runtime.memory is None:
            self.message("Memory is disabled for this session.")
            return
        browser = self.memory_browser
        while not browser.actions.empty():
            browser.actions.get_nowait()
        records = runtime.memory.records()
        self.dispatch(lambda: browser.open(records, argument))
        try:
            while not self.cancelled.is_set():
                try:
                    action, ident = browser.actions.get(timeout=1)
                except queue.Empty:
                    updated = runtime.memory.records()
                    if updated != records:
                        records = updated
                        self.dispatch(lambda snapshot=records: browser.set_records(snapshot))
                    continue
                if action == "close":
                    return
                if action == "delete":
                    try:
                        notice = runtime.forget_memory(ident, session)
                    except Exception as exc:
                        notice = "Deletion failed: " + str(exc)
                    records = runtime.memory.records()
                    self.dispatch(lambda snapshot=records, text=notice: browser.finish_delete(snapshot, text))
        finally:
            def hide():
                browser.active = False
                browser.pending_delete = None
                self.app.layout.focus(self.composer)
                self.app.invalidate()
            self.dispatch(hide)

    @property
    def active_browser(self):
        if self.memory_browser.active:
            return self.memory_browser
        if self.session_browser.active:
            return self.session_browser
        return None

    def show_session(self, session, state, previous, deleted=None):
        from .memory_view import safe_text
        if previous != deleted:
            self._session_views[previous] = (self.transcript, self.composer.text)
        if deleted:
            self._session_views.pop(deleted, None)
        if session in self._session_views:
            transcript, draft = self._session_views[session]
        else:
            transcript = f"Session {safe_text(session)}\n"
            for message in state.get("messages", []):
                if message['role'] == 'user':
                    transcript += '\n› ' + safe_text(message.get('content', '')) + '\n\n'
                elif message['role'] == 'assistant' and not message.get('tool_calls'):
                    transcript += safe_text(message.get('content', '')) + '\n'
            draft = ''
        self.transcript = transcript
        # Input history belongs to the displayed conversation too.
        self.composer.buffer.history = InMemoryHistory([
            m.get('content', '') for m in state.get('messages', []) if m['role'] == 'user'])
        self.composer.buffer.reset()
        self.composer.text = draft
        self.composer.buffer.cursor_position = len(draft)
        self._scroll_line = self._wrapped_key = None
        self.clear_selection()
        self.begin_turn()
        self.app.invalidate()

    def browse_sessions(self, runtime, session):
        from .sessions import list_sessions, new_session_id
        browser = self.session_browser
        while not browser.actions.empty():
            browser.actions.get_nowait()
        records = list_sessions(runtime.settings.state_dir)
        self.dispatch(lambda: browser.open(records, session))
        try:
            while not self.cancelled.is_set():
                try:
                    action, ident = browser.actions.get(timeout=1)
                except queue.Empty:
                    continue
                if action == 'close':
                    return session
                try:
                    if action == 'load':
                        state = runtime.load_session(ident)
                        old, session = session, ident
                        self.dispatch(lambda target=session, snapshot=state, previous=old:
                                      self.show_session(target, snapshot, previous))
                        return session
                    if action == 'delete':
                        runtime.delete_session(ident)
                        if ident == session:
                            old, session = session, new_session_id()
                            self.dispatch(lambda target=session, previous=old:
                                          self.show_session(target, {}, previous, deleted=previous))
                        else:
                            self.dispatch(lambda target=ident: self._session_views.pop(target, None))
                        notice = 'Session deleted.'
                    else:
                        continue
                except Exception as exc:
                    notice = 'Session operation failed: ' + str(exc)
                records = list_sessions(runtime.settings.state_dir)
                self.dispatch(lambda snapshot=records, current=session, text=notice:
                              browser.finish(snapshot, current, text))
            return session
        finally:
            def hide():
                browser.active = False
                browser.pending_delete = None
                browser.working = False
                self.app.layout.focus(self.composer)
                self.app.invalidate()
            self.dispatch(hide)

    def read_prompt(self):
        self._check_cancelled()

        def ready():
            self.busy = False
            self.status_message = ""
            self.app.invalidate()

        self.dispatch(ready)
        value = self._requests.get()
        if value is None:
            raise EOFError
        return value

    def approve(self, request, cwd, operation):
        self._check_cancelled()
        self.pause()
        suffix = " (UNSANDBOXED HOST COMMAND)" if operation == "run_cmd" else ""
        text = f"\n{operation}{suffix}\nWorking directory: {cwd}\nExact request: {request}\n"
        self.dispatch(lambda: setattr(self, "_transcript_limit", max(200_000, len(text) + 1024)))
        self.message(f"\033[36m{text}\033[0m" if self.color else text)

        def show():
            self.approval_pending = True
            self.composer.buffer.reset()
            self.status_message = "Permission required"
            self.app.invalidate()

        self.dispatch(show)
        return self._approvals.get()

    def _choose_command(self):
        command = self.matches[self.selected_command % len(self.matches)]
        self.composer.text = command.name + (" " if command.arguments else "")
        self.composer.buffer.cursor_position = len(self.composer.text)
        self._suppress_menu = True

    def submit(self):
        if self.active_browser:
            self.active_browser.enter()
            return
        if self.approval_pending:
            self._resolve_approval(self.composer.text.strip().lower() == "yes")
        elif self.busy:
            return
        elif self.menu_visible:
            self._choose_command()
        elif self.composer.text.strip():
            text = self.composer.text
            self.composer.buffer.append_to_history()
            self.composer.buffer.reset()
            self.busy = True
            self._scroll_line = None
            self.append(f"\n› {text}\n\n")
            self._requests.put(text)

    def _resolve_approval(self, allowed):
        self.approval_pending = False
        self.composer.buffer.reset()
        self.status_message = ""
        result = "approved" if allowed else "denied"
        self.append((f"\033[36mPermission {result}.\033[0m\n" if self.color else f"Permission {result}.\n"))
        self._approvals.put(allowed)
        self._transcript_limit = 200_000

    def _key_bindings(self):
        bindings = KeyBindings()

        @bindings.add("enter")
        def enter(event):
            self.submit()

        @bindings.add("tab", filter=Condition(lambda: self.menu_visible))
        def complete(event):
            self._choose_command()

        @bindings.add("up", filter=Condition(lambda: self.menu_visible))
        def previous(event):
            self.selected_command = (self.selected_command - 1) % len(self.matches)

        @bindings.add("down", filter=Condition(lambda: self.menu_visible))
        def next_command(event):
            self.selected_command = (self.selected_command + 1) % len(self.matches)

        @bindings.add("escape")
        def escape(event):
            if self.active_browser:
                self.active_browser.leave()
                return
            self.clear_selection()
            if self.approval_pending:
                self._resolve_approval(False)
            self._suppress_menu = True

        @bindings.add("c-j")
        def newline(event):
            if self.active_browser:
                return
            if not self.busy and not self.approval_pending:
                self.composer.buffer.insert_text("\n")

        @bindings.add("pageup")
        def pageup(event):
            if self.active_browser:
                if self.memory_browser.active:
                    self.memory_browser.page_details(-1)
                return
            self.scroll_conversation(-max(1, self.app.output.get_size().rows - 8))

        @bindings.add("pagedown")
        def pagedown(event):
            if self.active_browser:
                if self.memory_browser.active:
                    self.memory_browser.page_details(1)
                return
            self.scroll_conversation(max(1, self.app.output.get_size().rows - 8))

        @bindings.add("c-c")
        def interrupt(event):
            if self.active_browser:
                self.active_browser.leave()
                return
            if not self.approval_pending and self.copy_selection():
                return
            if self.approval_pending:
                self._resolve_approval(False)
            elif not self.busy:
                if self.composer.text:
                    self.composer.buffer.reset()
                else:
                    self.busy = True
                    self._requests.put(None)
            else:
                self.cancelled.set()
                self.status_message = "Stopping after the active operation returns…"

        @bindings.add("c-d")
        def eof(event):
            if self.active_browser:
                self.active_browser.leave()
                return
            if self.approval_pending:
                self._resolve_approval(False)
            elif not self.busy and not self.composer.text:
                self.busy = True
                self._requests.put(None)

        browser_active = Condition(lambda: self.active_browser is not None)
        memory_active = Condition(lambda: self.memory_browser.active)

        @bindings.add("up", filter=browser_active)
        def memory_previous(event):
            self.active_browser.move(-1)

        @bindings.add("down", filter=browser_active)
        def memory_next(event):
            self.active_browser.move(1)

        @bindings.add("tab", filter=browser_active)
        def memory_focus(event):
            self.active_browser.focus_next()

        @bindings.add("backspace", filter=Condition(lambda: self.active_browser is not None
                      and self.app.layout.has_focus(self.active_browser.cards)
                      and not self.active_browser.pending_delete))
        def memory_delete(event):
            self.active_browser.ask_delete()

        @bindings.add("escape", "m", filter=memory_active)
        def memory_modify(event):
            self.memory_browser.leave(modify=True)

        return bindings

    async def run_async(self, callback):
        def start():
            self._loop = asyncio.get_running_loop()
            self._ui_thread = threading.get_ident()

            def worker():
                try:
                    result = callback()
                except KeyboardInterrupt:
                    self.message("Interrupted. Inspect any pending host command's effects before retrying.")
                    result = 130
                except Exception as exc:
                    self.message(f"Harness: {exc}")
                    result = 1
                def finish():
                    if self.app.is_running and not self.app.is_done:
                        self.app.exit(result=result)

                self.dispatch(finish)

            self._worker = threading.Thread(target=worker, name="harness-runtime", daemon=True)
            self._worker.start()

        try:
            return await self.app.run_async(pre_run=start)
        finally:
            self.cancelled.set()
            self._requests.put(None)
            self._approvals.put(False)
            if self._worker is not None:
                await asyncio.to_thread(self._worker.join)
            self._loop = None

    def run(self, callback):
        result = asyncio.run(self.run_async(callback))
        # Preserve the session in normal terminal scrollback after leaving the alternate screen.
        transcript = self.transcript if self.color else re.sub(r"\033\[[0-9;]*m", "", self.transcript)
        print(transcript, end="" if transcript.endswith("\n") else "\n", flush=True)
        return result
