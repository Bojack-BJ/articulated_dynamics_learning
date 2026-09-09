#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_DIR="${CODEX_PRESENTATIONS_SKILL_DIR:-$HOME/.codex/plugins/cache/openai-primary-runtime/presentations/26.730.11710/skills/presentations}"
WORKSPACE="$ROOT/.tmp/track2art_method_overview"

mkdir -p "$WORKSPACE"
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" --workspace "$WORKSPACE"
cp "$ROOT/scripts/generate_track2art_method_overview.mjs" "$WORKSPACE/build.mjs"
cp "$ROOT/scripts/combine_track2art_method_overview_versions.mjs" "$WORKSPACE/combine.mjs"
cd "$ROOT"
node "$WORKSPACE/build.mjs"
node "$WORKSPACE/combine.mjs"

if command -v soffice >/dev/null 2>&1; then
  mkdir -p /tmp/lo-track2art
  soffice '-env:UserInstallation=file:///tmp/lo-track2art' --headless \
    --convert-to pdf --outdir "$ROOT/outputs" \
    "$ROOT/outputs/track2art_method_overview.pptx" >/dev/null
fi

echo "$ROOT/outputs/track2art_method_overview.pptx"
echo "$ROOT/outputs/track2art_method_overview_preview.png"
test ! -f "$ROOT/outputs/track2art_method_overview.pdf" || \
  echo "$ROOT/outputs/track2art_method_overview.pdf"
