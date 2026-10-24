"""Saved-session cards with search, resume and confirmed deletion."""

import queue
import textwrap

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import DynamicContainer, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Box, TextArea

from .browser import CardControl
from .memory_view import age, safe_text


class SessionCards(CardControl):
    def fragments(self):
        browser = self.browser
        self.line_indices = []
        self.cursor_row = 0
        fragments = []
        if not browser.matches:
            return [('class:muted', '\n  No saved sessions match. Try different keywords.\n')]
        inner = max(1, self.width - 4)
        for index, record in enumerate(browser.matches):
            selected = index == browser.index
            if selected:
                self.cursor_row = len(self.line_indices)
            current = ' · CURRENT' if record['id'] == browser.current else ''
            title = f" {'>' if selected else ' '} SESSION{current} "
            lines = ['╭' + title[:self.width - 2].ljust(self.width - 2, '─') + '╮']
            content = textwrap.wrap(safe_text(record.get('name', 'UNKNOWN')), inner) or ['UNKNOWN']
            content += [safe_text(record['id']),
                        f"{age(record.get('updated_at'))} · {record['messages']} retained messages"]
            lines += ['│ ' + line[:inner].ljust(inner) + ' │' for line in content]
            lines += ['╰' + '─' * (self.width - 2) + '╯', '']
            for number, line in enumerate(lines):
                style = 'class:memory-selected' if selected else ('class:memory-heading' if number == 0 else '')
                fragments.append((style, line + '\n'))
                self.line_indices.append(index)
        return fragments


class SessionBrowser:
    def __init__(self, ui):
        self.ui = ui
        self.active = False
        self.current = None
        self.records, self.matches = [], []
        self.index = 0
        self.actions = queue.Queue()
        self.pending_delete = None
        self.working = False
        self.notice = ''
        self.cards = SessionCards(self)
        self.cards_window = Window(self.cards, always_hide_cursor=True)
        self.search = TextArea(multiline=False, prompt='Search session names › ', height=1,
                               read_only=Condition(lambda: self.working or self.pending_delete is not None))
        self.search.buffer.on_text_changed += lambda _: self.filter()
        self.confirmation = TextArea(multiline=False, height=1, prompt='Type yes to delete › ')
        self.container = HSplit([
            Window(FormattedTextControl(' Agent chat  |  [ Sessions ]'), height=2, style='class:memory-heading'),
            Window(FormattedTextControl(lambda: f' Sessions  {len(self.matches)}/{len(self.records)}'),
                   height=1, style='class:memory-heading'),
            self.cards_window,
            Window(FormattedTextControl(lambda: self.notice), height=2, style='class:approval'),
            Box(DynamicContainer(lambda: self.confirmation if self.pending_delete else self.search),
                padding_top=1, padding_bottom=1, padding_left=1, padding_right=1,
                style=lambda: 'class:approval-input' if self.pending_delete else 'class:composer'),
            Window(FormattedTextControl(self.footer), height=2, style='class:muted'),
        ])

    @property
    def selected(self):
        return self.matches[self.index] if self.matches else None

    def set_records(self, records):
        selected = self.selected['id'] if self.selected else None
        self.records = records
        self.filter(selected)

    def filter(self, selected=None):
        selected = selected or (self.selected['id'] if self.selected else None)
        words = self.search.text.casefold().split()
        self.matches = [item for item in self.records
                        if all(word in item.get('name', 'UNKNOWN').casefold() for word in words)]
        identifiers = [item['id'] for item in self.matches]
        self.index = identifiers.index(selected) if selected in identifiers else min(self.index, max(0, len(identifiers) - 1))
        self.ui.app.invalidate()

    def open(self, records, current):
        self.active, self.working = True, False
        self.current, self.pending_delete, self.notice = current, None, ''
        self.set_records(records)
        self.ui.app.layout.focus(self.cards)
        self.ui.app.invalidate()

    def select(self, index):
        if not self.pending_delete and not self.working:
            self.index = max(0, min(index, len(self.matches) - 1))
            self.ui.app.invalidate()

    def move(self, delta):
        self.select(self.index + delta)

    def focus_next(self):
        if not self.pending_delete and not self.working:
            self.ui.app.layout.focus(self.cards if self.ui.app.layout.has_focus(self.search) else self.search)

    def leave(self):
        if self.working:
            return
        self.active, self.pending_delete = False, None
        self.ui.app.layout.focus(self.ui.composer)
        self.actions.put(('close', None))
        self.ui.app.invalidate()

    def ask_delete(self):
        if self.selected and not self.pending_delete and not self.working:
            self.pending_delete = self.selected['id']
            self.confirmation.text = ''
            self.notice = 'Delete this saved conversation? Only yes confirms. Esc returns to chat.'
            self.ui.app.layout.focus(self.confirmation)
            self.ui.app.invalidate()

    def enter(self):
        if self.working:
            return
        if self.pending_delete:
            ident = self.pending_delete
            confirmed = self.confirmation.text.strip().lower() == 'yes'
            self.pending_delete = None
            self.ui.app.layout.focus(self.cards)
            if confirmed:
                self.working, self.notice = True, 'Deleting session…'
                self.actions.put(('delete', ident))
            else:
                self.notice = 'Deletion cancelled.'
        elif self.selected:
            self.working, self.notice = True, 'Loading session…'
            self.actions.put(('load', self.selected['id']))
        self.ui.app.invalidate()

    def finish(self, records, current, notice):
        self.working, self.current, self.notice = False, current, notice
        self.set_records(records)

    def footer(self):
        key = 'class:muted bold' if self.ui.color else 'class:muted'
        return [(key, ' ↑/↓'), ('class:muted', ' select · '), (key, 'Enter'), ('class:muted', ' load · '),
                (key, 'Tab'), ('class:muted', ' search/cards · '), (key, 'Backspace'),
                ('class:muted', ' delete\n '), (key, 'Esc'), ('class:muted', ' agent chat · Keyword name search')]
