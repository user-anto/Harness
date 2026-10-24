"""Interactive memory browser: cards, complete details, keyword search and actions."""

import json
import queue
from pathlib import Path
import textwrap

from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Box, TextArea

from .browser import CardControl
from .memory_view import LABELS, age, safe_text


class MemoryCards(CardControl):
    def render_card(self, record, selected):
        label, color = LABELS.get(record.get('kind'), LABELS['user_stated'])
        title = f" {'>' if selected else ' '} {label} | {record['id'][:8]} "
        inner = self.width - 4
        lines = ['╭' + title[:self.width - 2].ljust(self.width - 2, '─') + '╮']
        for line in textwrap.wrap(safe_text(record['statement']), inner)[:3] or ['']:
            lines.append('│ ' + line.ljust(inner) + ' │')
        metadata = age(record.get('updated_at'))
        lines.append('│ ' + safe_text(metadata)[:inner].ljust(inner) + ' │')
        lines.extend(['╰' + '─' * (self.width - 2) + '╯', ''])
        result = []
        for number, line in enumerate(lines):
            if self.browser.ui.color and not selected and number in (0, len(lines) - 2):
                result.append(list(to_formatted_text(ANSI(f'\033[{color}m{line}\033[0m'))))
            else:
                result.append([('class:memory-selected' if selected else '', line)])
        return result


class MemoryBrowser:
    def __init__(self, ui):
        self.ui = ui
        self.active = False
        self.records = []
        self._search_text = {}
        self.matches = []
        self.index = 0
        self.actions = queue.Queue()
        self.pending_delete = None
        self.working = False
        self.notice = ''
        self.cards = MemoryCards(self)
        self.cards_window = Window(self.cards, always_hide_cursor=True)
        self.details = TextArea(read_only=True, scrollbar=True, wrap_lines=True, focus_on_click=True)
        self.search = TextArea(multiline=False, prompt='Search memories › ', height=1,
                               read_only=Condition(lambda: self.working or self.pending_delete is not None))
        self.search.buffer.on_text_changed += lambda _: self.filter()
        self.confirmation = TextArea(multiline=False, height=1, prompt='Type yes to delete › ')
        self.left = HSplit([Window(FormattedTextControl(lambda: f' Memories  {len(self.matches)}/{len(self.records)}'),
                                         height=1, style='class:memory-heading'), self.cards_window],
                          width=lambda: max(1, (ui.app.output.get_size().columns - 1) * 7 // 10))
        self.right = Box(HSplit([Window(FormattedTextControl('Full details'), height=1,
                                   style='class:memory-heading'), self.details]),
                         padding=0, padding_left=2, padding_right=2,
                         width=lambda: max(1, (ui.app.output.get_size().columns - 1) -
                                           (ui.app.output.get_size().columns - 1) * 7 // 10))
        from prompt_toolkit.layout import DynamicContainer
        self.container = HSplit([
            Window(FormattedTextControl(' Agent chat  |  [ Memories ]'), height=2, style='class:memory-heading'),
            VSplit([self.left, Window(width=1, char='│', style='class:muted'), self.right]),
            Window(FormattedTextControl(self.status), height=2, style='class:approval'),
            Box(DynamicContainer(lambda: self.confirmation if self.pending_delete else self.search),
                padding_top=1, padding_bottom=1, padding_left=1, padding_right=1,
                style=lambda: 'class:approval-input' if self.pending_delete else 'class:composer'),
            Window(FormattedTextControl(self.footer), height=2, style='class:muted'),
        ])

    @property
    def selected(self):
        return self.matches[self.index] if self.matches else None

    def set_records(self, records):
        selected_id = self.selected['id'] if self.selected else None
        self.records = sorted(records, key=lambda item: (item.get('updated_at', ''), item['id']), reverse=True)
        self._search_text = {record["id"]: json.dumps(record, ensure_ascii=False).casefold() for record in self.records}
        self.filter(selected_id)

    def filter(self, selected_id=None):
        selected_id = selected_id or (self.selected['id'] if self.selected else None)
        words = self.search.text.casefold().split()
        self.matches = [record for record in self.records
                        if all(word in self._search_text[record["id"]] for word in words)]
        identifiers = [record['id'] for record in self.matches]
        self.index = identifiers.index(selected_id) if selected_id in identifiers else min(self.index, max(0, len(identifiers) - 1))
        self.show_details()

    def show_details(self):
        record = self.selected
        if record:
            lines = ['STATEMENT', safe_text(record['statement']), '']
            for key, value in record.items():
                if key != 'statement':
                    if key == 'workspace' and isinstance(value, str):
                        home = str(Path.home())
                        if value == home or value.startswith(home + '/'):
                            value = '~' + value[len(home):]
                    lines.append(key.replace('_', ' ').upper())
                    lines.append(safe_text(json.dumps(value, indent=2, ensure_ascii=False)
                                           if isinstance(value, (list, dict)) else value))
                    lines.append('')
            text = '\n'.join(lines)
        else:
            text = 'Select a memory to see its complete details.'
        if self.details.text != text:
            self.details.text = text
        self.ui.app.invalidate()

    def select(self, index):
        if not self.pending_delete and not self.working:
            self.index = max(0, min(index, len(self.matches) - 1))
            self.show_details()

    def move(self, delta):
        self.select(self.index + delta)

    def open(self, records, argument=''):
        self.active = True
        self.pending_delete = None
        self.working = False
        self.notice = ''
        self.set_records(records)
        kinds = {'saved': 'explicit', 'facts': 'user_stated', 'inferred': 'inferred', 'notes': 'answer_note'}
        if argument in kinds:
            self.search.text = kinds[argument]
        elif argument:
            self.search.text = ''
            selected = next((index for index, item in enumerate(self.matches) if item['id'].startswith(argument)), None)
            if selected is not None:
                self.select(selected)
        self.ui.app.layout.focus(self.cards)
        self.ui.app.invalidate()

    def leave(self, modify=False):
        if modify and (not self.selected or self.pending_delete or self.working):
            return
        if modify:
            self.ui.composer.text = f"Modify memory {self.selected['id']}: "
            self.ui.composer.buffer.cursor_position = len(self.ui.composer.text)
        self.active = False
        self.pending_delete = None
        self.ui.app.layout.focus(self.ui.composer)
        self.actions.put(('close', None))
        self.ui.app.invalidate()

    def focus_next(self):
        if self.pending_delete or self.working:
            return
        focused = self.ui.app.layout.has_focus(self.search)
        self.ui.app.layout.focus(self.cards if focused else self.search)

    def ask_delete(self):
        if not self.selected or self.working or self.pending_delete:
            return
        self.pending_delete = self.selected['id']
        self.confirmation.text = ''
        self.notice = f"Delete {self.pending_delete[:8]} and all its versions? Only yes confirms. Esc returns to chat."
        self.ui.app.layout.focus(self.confirmation)
        self.ui.app.invalidate()

    def enter(self):
        if self.pending_delete:
            ident = self.pending_delete
            confirmed = self.confirmation.text.strip().lower() == 'yes'
            self.pending_delete = None
            self.ui.app.layout.focus(self.cards)
            if confirmed:
                self.working = True
                self.notice = 'Deleting memory…'
                self.actions.put(('delete', ident))
            else:
                self.notice = 'Deletion cancelled.'
        else:
            self.ui.app.layout.focus(self.cards)
        self.ui.app.invalidate()

    def finish_delete(self, records, notice):
        self.working = False
        self.notice = notice
        self.set_records(records)

    def page_details(self, delta):
        if self.pending_delete or self.working:
            return
        self.ui.app.layout.focus(self.details)
        rows = max(1, self.ui.app.output.get_size().rows - 10)
        if delta < 0:
            self.details.buffer.cursor_up(rows)
        else:
            self.details.buffer.cursor_down(rows)

    def status(self):
        return self.notice

    def footer(self):
        key = 'class:muted bold' if self.ui.color else 'class:muted'
        return [('class:muted', ' '), (key, '↑/↓'), ('class:muted', ' select · '),
                (key, 'Tab'), ('class:muted', ' search/cards · '),
                (key, 'Backspace'), ('class:muted', ' delete · '),
                (key, 'Alt+M'), ('class:muted', ' modify\n '),
                (key, 'Esc'), ('class:muted', ' agent chat · '),
                (key, 'PgUp/PgDn'), ('class:muted', ' details · Keyword search')]
