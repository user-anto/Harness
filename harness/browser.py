"""Shared lazy card control: layout once, render only requested viewport lines."""

from bisect import bisect_right
import time

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
from prompt_toolkit.mouse_events import MouseEventType


class CardControl(FormattedTextControl):
    def __init__(self, browser):
        self.browser = browser
        self.width = 60
        self.cursor_row = 0
        self._layout_key = None
        self._offsets = []
        self._normal = []
        self._selected = None
        super().__init__(focusable=True)

    def create_content(self, width, height):
        self.width = max(8, width)
        records = self.browser.matches
        key = (id(records), self.width, int(time.time() // 60), getattr(self.browser, "current", None))
        if key != self._layout_key:
            self._normal = [self.render_card(record, False) for record in records]
            self._offsets = [0]
            for lines in self._normal:
                self._offsets.append(self._offsets[-1] + len(lines))
            self._selected = None
            self._layout_key = key
        if not records:
            self.cursor_row = 0
            label = "sessions" if hasattr(self.browser, "current") else "memories"
            return UIContent(get_line=lambda row: [("class:muted", f"  No {label} match. Try different keywords.")]
                             if row == 1 else [], line_count=3, cursor_position=Point(0, 0))
        index = self.browser.index
        selected_key = (key, index)
        if self._selected is None or self._selected[0] != selected_key:
            self._selected = (selected_key, self.render_card(records[index], True))
        self.cursor_row = self._offsets[index]

        def get_line(row):
            card = bisect_right(self._offsets, row) - 1
            if not 0 <= card < len(records):
                return []
            lines = self._selected[1] if card == index else self._normal[card]
            return lines[row - self._offsets[card]]

        return UIContent(get_line=get_line, line_count=self._offsets[-1],
                         cursor_position=Point(0, self.cursor_row), show_cursor=False)

    def mouse_handler(self, event):
        if event.event_type == MouseEventType.SCROLL_UP:
            self.browser.move(-1)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self.browser.move(1)
        elif event.event_type == MouseEventType.MOUSE_DOWN and self.browser.matches:
            self.browser.select(max(0, min(len(self.browser.matches) - 1,
                                          bisect_right(self._offsets, event.position.y) - 1)))
            self.browser.ui.app.layout.focus(self)
        else:
            return NotImplemented
        return None
