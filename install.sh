#!/bin/bash
set -e

# Find the absolute path to this repository
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target directory for the executable
BIN_DIR="$HOME/.local/bin"

# Ensure the bin directory exists
mkdir -p "$BIN_DIR"

# Create the harness executable wrapper
cat << EOF > "$BIN_DIR/harness"
#!/bin/bash
# Automatically generated wrapper for Harness
export PYTHONPATH="\$PYTHONPATH:$REPO_DIR/src"

# Prefer the virtual environment's python if it exists
if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    exec "$REPO_DIR/.venv/bin/python" "$REPO_DIR/src/graph.py" "\$@"
else
    exec python3 "$REPO_DIR/src/graph.py" "\$@"
fi
EOF

chmod +x "$BIN_DIR/harness"

# Check if llama-server is installed
if ! command -v llama-server &> /dev/null; then
    echo "WARNING: 'llama-server' (llama.cpp) was not found in your PATH."
    echo "Please ensure llama-server is installed and in your PATH to run the local orchestrator."
fi

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "WARNING: 'ollama' was not found in your PATH."
    echo "Please ensure Ollama is installed and running locally."
fi

echo "'harness' command installed to $BIN_DIR/harness"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "WARNING: $BIN_DIR is not in your PATH."
    echo "Please add the following line to your ~/.bashrc or ~/.zshrc:"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "Then restart your terminal."
else
    echo "You can now use the 'harness' command from anywhere!"
fi
