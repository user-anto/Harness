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
cd "$REPO_DIR/src" || exit 1

# Prefer the virtual environment's python if it exists
if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    exec "$REPO_DIR/.venv/bin/python" orch.py "\$@"
else
    exec python3 orch.py "\$@"
fi
EOF

chmod +x "$BIN_DIR/harness"

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
