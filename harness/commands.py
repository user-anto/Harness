"""Commands available in the interactive composer and plain CLI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    arguments: str
    description: str


COMMANDS = (
    Command("/plan", "GOAL", "Propose a plan before execution"),
    Command("/remember", "FACT", "Save a fact to long-term memory"),
    Command("/memories", "", "Browse memories; add notes for answer notes or ID for evidence"),
    Command("/sessions", "", "Browse saved sessions, search, resume or delete"),
    Command("/clear", "", "Clear this session's active context"),
    Command("/help", "", "Show available commands"),
    Command("/bye", "", "Close this Harness session"),
)


def command_help():
    return "\n".join(f"{command.name} {command.arguments}".rstrip() + f"  — {command.description}"
                     for command in COMMANDS)
