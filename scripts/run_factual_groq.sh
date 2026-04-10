#!/usr/bin/env bash
#
# Run all Groq factual tests against a freshly downloaded test image.
#
# The image is downloaded to a temp path, used by every test, then removed
# (even if a test fails) via an EXIT trap.
#
# Usage:
#   ./scripts/run_factual_groq.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_URL="${TEST_IMAGE_URL:-https://picsum.photos/seed/aiyer/768/768.jpg}"
IMAGE_PATH="$(mktemp --suffix=.jpg)"

cleanup() {
    rm -f "$IMAGE_PATH"
}
trap cleanup EXIT

echo "Downloading test image from: $IMAGE_URL"
curl -fsSL -o "$IMAGE_PATH" "$IMAGE_URL"
echo "Saved to: $IMAGE_PATH ($(wc -c < "$IMAGE_PATH") bytes)"
echo

TESTS=(
    test_zero
    test_lite
    test_medium
    test_chat
)

failed=0
for t in "${TESTS[@]}"; do
    echo "=============================================="
    echo " Running tests.factual.groq.$t"
    echo "=============================================="
    if ! python -m "tests.factual.groq.$t" "$IMAGE_PATH"; then
        echo "FAILED: tests.factual.groq.$t"
        failed=$((failed + 1))
    fi
    echo
done

if [ "$failed" -ne 0 ]; then
    echo "$failed test(s) failed."
    exit 1
fi

echo "All Groq factual tests passed."
