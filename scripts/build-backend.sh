#!/bin/bash
set -e

echo "Building whisper-subtitler API backend..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Creating temporary build environment..."
if [ ! -d ".venv" ]; then
    uv venv
fi

echo "Installing PyInstaller and dependencies..."
source .venv/bin/activate
uv pip install pyinstaller

echo "Ensuring project dependencies are installed..."
uv pip install -e .

echo "Running PyInstaller..."
pyinstaller api.spec

deactivate

echo "Build complete!"
echo "Binary location: dist/"

ls -lh dist/

BINARY_NAME=$(ls dist/ | head -1)
echo ""
echo "To test the binary:"
echo "  ./dist/${BINARY_NAME}"
echo ""
echo "To copy to Tauri:"
echo "  mkdir -p whisper-subtitler-app/src-tauri/bin"
echo "  cp dist/${BINARY_NAME} whisper-subtitler-app/src-tauri/bin/"
echo "  chmod +x whisper-subtitler-app/src-tauri/bin/*"
