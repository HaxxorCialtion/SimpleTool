#!/bin/bash
set -euo pipefail
infra_root="$(cd "$(dirname "$0")/.." && pwd)"
infra_build="${1:-/tmp/simplelove-infra-xcode-build}"
cd "$infra_root/SimpleToolSwift"
xcodebuild -scheme InfraSmoke -destination 'platform=macOS' -derivedDataPath "$infra_build" -jobs 4 build
printf '%s\n' "Run: $infra_build/Build/Products/Debug/InfraSmoke MODEL_DIRECTORY RECEIPT.json"
