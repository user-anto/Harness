# Terminal input

## Selection and scrolling

Drag with the left mouse button over conversation text to select it. The selected
range is highlighted, Ctrl+C copies it without clearing your draft or interrupting
generation, and Esc clears the selection. Soft line wraps are removed when copying;
actual paragraph breaks remain. Wheel/trackpad scrolling still moves the conversation.
Text in the composer retains its normal text-editing behavior.

Copy goes to Harness's internal clipboard (Ctrl+Y to paste into the composer) and
to the terminal's system clipboard through OSC 52. If your terminal disables OSC 52,
enable it or use Shift+drag for native terminal selection where supported. Permission
prompts keep their existing Ctrl+C-to-deny behavior.

## Memories tab

`/memories` opens the split view inside the same terminal: 70% cards on the left,
30% complete details on the right. Up/Down chooses a card and refreshes its details.
Tab changes keyboard focus between the cards and live keyword search. Enter in the
search box focuses the results. Search uses no LLM or embedding request.

Backspace while the cards are focused asks for confirmation; type `yes` and Enter
to delete all versions of that fact. Any other answer cancels. Backspace in the
search box remains ordinary text editing. Confirmation input and notices are cyan.
Esc always returns to agent chat, cancelling an unconfirmed deletion. Search and
selection are retained when reopening. Alt+M returns to chat and inserts
`Modify memory FULL_ID: ` for the selected card, ready for editing instructions.
PgUp/PgDn scrolls the full details; clicking the details pane permits mouse scrolling.

## Sessions tab

`/sessions` opens a single full-width list of saved-session cards. Up/Down changes
selection. Tab moves focus between cards and live keyword search; only session
display names are searched. Enter loads the selected session, displays its retained
conversation and restores its workspace. Esc returns to the current chat without
switching. Search and selection survive closing and reopening the tab.

Backspace on a card opens a cyan confirmation prompt; only `yes` and Enter deletes
it. Backspace while searching edits the query. Deleting the current session creates
a fresh chat; global memories remain available. Shortcut labels are bold in both
browser footers. Workspace paths are omitted from cards.

New session names are shortened first-prompt titles, with no extra model request.
Older sessions display `UNKNOWN`. IDs remain distinct and stable for `--session`.

## Newlines

Enter sends a prompt. Shift+Enter inserts a newline in the composer, and Ctrl+J
provides a fallback. The application does not bind Alt+Enter; your terminal may use
it to toggle fullscreen.

Harness recognizes Shift+Enter encoded as CSI-u (`ESC [ 13 ; 2 u`) or xterm
modifyOtherKeys (`ESC [ 27 ; 2 ; 13 ~`). Left and right Shift are not distinguished
by these encodings. A terminal that sends the same carriage return for Enter and
Shift+Enter must be configured to send a distinct sequence.

## Windows Terminal

Open Settings → Open JSON file. Add the following entry to the existing `actions`
array, preserving the other entries:

```json
{ "command": { "action": "sendInput", "input": "\u001b[13;2u" }, "keys": "shift+enter" }
```

If `shift+enter` already has a binding, replace that binding. This applies to both
Shift keys. Reload the terminal if the mapping is not picked up immediately.
