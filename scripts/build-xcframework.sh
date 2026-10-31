#!/usr/bin/env bash
# Build a static in-process macOS library. No recognition model is embedded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${1:?usage: build-xcframework.sh /absolute/output/OcrusFFI.xcframework}"
[[ "$OUTPUT" = /* && "$OUTPUT" = *.xcframework ]] || { echo 'Use an absolute .xcframework output path' >&2; exit 1; }
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cd "$ROOT"
RUN=()
if command -v mise >/dev/null 2>&1; then RUN=(mise exec --); fi
# Use rustup's compiler explicitly: a Homebrew compiler earlier on PATH can
# otherwise pull in a standard library built only for the developer's OS.
toolchain="$("${RUN[@]}" rustup show active-toolchain | cut -d ' ' -f 1)"
compiler="$("${RUN[@]}" rustup which --toolchain "$toolchain" rustc)"
ARCHIVES=()
for arch in ${OCRUS_ARCHS:-$(uname -m)}; do
  case "$arch" in
    arm64|aarch64) target=aarch64-apple-darwin ;;
    x86_64) target=x86_64-apple-darwin ;;
    *) echo "Unsupported macOS architecture: $arch" >&2; exit 1 ;;
  esac
  "${RUN[@]}" rustup target add --toolchain "$toolchain" "$target"
  MACOSX_DEPLOYMENT_TARGET=11.0 RUSTC="$compiler" "${RUN[@]}" rustup run "$toolchain" cargo build --locked --release -p ocrus-ffi --features accelerate --target "$target"
  target_dir="$("${RUN[@]}" cargo metadata --no-deps --format-version 1 | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
  ARCHIVES+=("$target_dir/$target/release/libocrus_ffi.a")
done
mkdir -p "$STAGE/include"
cp crates/ocrus-ffi/include/ocrus.h "$STAGE/include/"
cat > "$STAGE/include/module.modulemap" <<'EOF'
module OcrusFFI {
  header "ocrus.h"
  export *
}
EOF
lipo -create "${ARCHIVES[@]}" -output "$STAGE/libOcrusFFI.a"
mkdir -p "$(dirname "$OUTPUT")"
rm -rf "$OUTPUT"
xcodebuild -create-xcframework -library "$STAGE/libOcrusFFI.a" -headers "$STAGE/include" -output "$OUTPUT"
