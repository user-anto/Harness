"""Shared focusable card control for terminal browsers."""

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType


class CardControl(FormattedTextControl):
    def __init__(self, browser):
        self.browser = browser
        self.width = 60
        self.line_indices = []
        self.cursor_row = 0
        super().__init__(self.fragments, focusable=True,
                         get_cursor_position=lambda: Point(0, self.cursor_row))

    def create_content(self, width, height):
        self.width = max(8, width)
        return super().create_content(width, height)

    def mouse_handler(self, event):
        if event.event_type == MouseEventType.SCROLL_UP:
            self.browser.move(-1)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self.browser.move(1)
        elif event.event_type == MouseEventType.MOUSE_DOWN and self.line_indices:
            self.browser.select(self.line_indices[min(event.position.y, len(self.line_indices) - 1)])
            self.browser.ui.app.layout.focus(self)
        else:
            return NotImplemented
        return None
