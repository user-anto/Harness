from graph import app
from cli import render_terminal_ui

if __name__ == "__main__":
    render_terminal_ui()

    response = app.invoke({})