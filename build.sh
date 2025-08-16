#! /bin/bash
set -euo pipefail

# Build Game Data Collector binaries
# Usage:
#   ./build.sh mac       # Build macOS binary locally
#   ./build.sh windows   # Build Windows .exe (native if on Windows; otherwise via Docker if available)
#   ./build.sh all       # Build both (mac locally + windows via Docker if not on Windows)

APP_NAME="Game_Data_Collector_V1"
ENTRY="src/streaming/android_streamer.py"
DIST_DIR="dist"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"

need_pyinstaller() {
  if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "➡️  Installing PyInstaller..."
    python3 -m pip install --user pyinstaller
    export PATH="$HOME/Library/Python/*/bin:$HOME/.local/bin:$PATH"
  fi
}

build_mac() {
  echo "🛠️  Building macOS binary..."
  need_pyinstaller
  pyinstaller --noconfirm --onefile --console --name "$APP_NAME" "$ENTRY"
  mkdir -p "$DIST_DIR/mac"
  if [ -f "$DIST_DIR/$APP_NAME" ]; then
    mv -f "$DIST_DIR/$APP_NAME" "$DIST_DIR/mac/$APP_NAME"
  elif [ -f "$DIST_DIR/$APP_NAME/$APP_NAME" ]; then
    mv -f "$DIST_DIR/$APP_NAME/$APP_NAME" "$DIST_DIR/mac/$APP_NAME"
  fi
  echo "✅ macOS binary: $DIST_DIR/mac/$APP_NAME"
}

build_windows_native() {
  echo "🛠️  Building Windows binary (native)..."
  need_pyinstaller
  pyinstaller --noconfirm --onefile --console --name "$APP_NAME" "$ENTRY"
  mkdir -p "$DIST_DIR/windows"
  if [ -f "$DIST_DIR/$APP_NAME.exe" ]; then
    mv -f "$DIST_DIR/$APP_NAME.exe" "$DIST_DIR/windows/$APP_NAME.exe"
  elif [ -f "$DIST_DIR/$APP_NAME/$APP_NAME.exe" ]; then
    mv -f "$DIST_DIR/$APP_NAME/$APP_NAME.exe" "$DIST_DIR/windows/$APP_NAME.exe"
  fi
  echo "✅ Windows binary: $DIST_DIR/windows/$APP_NAME.exe"
}

build_windows_docker() {
  echo "🛠️  Building Windows binary via Docker (cross-compile)..."
  if [ "$OS" = "darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "❌ Windows cross-build via Docker/Wine is unreliable on Apple Silicon (arm64)."
    echo "➡️  Use the GitHub Actions workflow: .github/workflows/build-windows.yml"
    echo "    Trigger it from the Actions tab to produce a native Windows .exe artifact."
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found. Install Docker Desktop or run this on a Windows machine."
    exit 1
  fi
  # Use a PyInstaller-for-Windows image with Wine. This pulls if missing.
  # Note: On Apple Silicon, Docker may emulate linux/amd64, which is slower.
  docker run --rm \
    -v "$(pwd)":/src \
    -w /src \
    --platform linux/amd64 \
    cdrx/pyinstaller-windows \
    "pyinstaller --noconfirm --onefile --console --name $APP_NAME $ENTRY"

  mkdir -p "$DIST_DIR/windows"
  if [ -f "$DIST_DIR/$APP_NAME.exe" ]; then
    mv -f "$DIST_DIR/$APP_NAME.exe" "$DIST_DIR/windows/$APP_NAME.exe"
  elif [ -f "$DIST_DIR/$APP_NAME/$APP_NAME.exe" ]; then
    mv -f "$DIST_DIR/$APP_NAME/$APP_NAME.exe" "$DIST_DIR/windows/$APP_NAME.exe"
  fi
  echo "✅ Windows binary: $DIST_DIR/windows/$APP_NAME.exe"
}

TARGET=${1:-mac}

case "$TARGET" in
  mac)
    build_mac
    ;;
  windows)
    if [[ "$OS" == *"mingw"* || "$OS" == *"cygwin"* || "$OS" == *"msys"* ]]; then
      build_windows_native
    else
      build_windows_docker
    fi
    ;;
  all)
    if [[ "$OS" == "darwin" ]]; then
      build_mac
      if [[ "$ARCH" == "arm64" ]]; then
        echo "⚠️  Skipping Windows Docker cross-build on Apple Silicon."
        echo "➡️  Use GitHub Actions workflow '.github/workflows/build-windows.yml' to build on Windows."
      else
        build_windows_docker || true
      fi
    elif [[ "$OS" == "linux" ]]; then
      echo "ℹ️ On Linux: building Windows via Docker only."
      build_windows_docker
    else
      echo "ℹ️ On Windows: building both natively."
      build_windows_native
      echo "⚠️ macOS build requires macOS. Skipping."
    fi
    ;;
  *)
    echo "Usage: $0 [mac|windows|all]"
    exit 1
    ;;
esac