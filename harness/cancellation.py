"""A recoverable interruption of the active turn, not the CLI session."""


class TurnInterrupted(Exception):
    pass
