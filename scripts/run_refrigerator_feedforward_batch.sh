#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

BATCH_CONFIG="${BATCH_CONFIG:-configs/batch_lightwheel_refrigerators_mjcf.tsv}"
EPISODE_ROOT="${EPISODE_ROOT:-outputs/recordings_refrigerators_staged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/feedforward_articulation/lightwheel_refrigerators_view1_dooropen_upY}"
REMOTE_ARTICULATION_SERVER_URL="${REMOTE_ARTICULATION_SERVER_URL:-http://127.0.0.1:8888}"
REMOTE_ARTICULATION_API_TOKEN="${REMOTE_ARTICULATION_API_TOKEN:-lxt}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"

mkdir -p "$OUTPUT_ROOT/_logs"
rm -f "$OUTPUT_ROOT/_logs/failures.txt"

if ! curl -fsS -m 10 -H "Authorization: Bearer ${REMOTE_ARTICULATION_API_TOKEN}" \
  "${REMOTE_ARTICULATION_SERVER_URL}/health" >/dev/null; then
  echo "Remote articulation server is not reachable: ${REMOTE_ARTICULATION_SERVER_URL}" >&2
  echo "Start the tunnel first, for example:" >&2
  echo "  ssh -N -L 8888:127.0.0.1:8090 root@dev-h" >&2
  exit 1
fi

failures=()
while IFS=$'\t' read -r category model_path object_id joint_name; do
  case "$category" in "#"*|"") continue ;; esac

  episode="${EPISODE_ROOT}/${object_id}/episode.json"
  out_dir="${OUTPUT_ROOT}/${object_id}"
  input_dir="${out_dir}/hunyuan_inputs"
  log="${OUTPUT_ROOT}/_logs/${object_id}.log"

  if [ -f "${out_dir}/particulate/particulate_result.json" ]; then
    echo "[$object_id] skip existing ${out_dir}/particulate/particulate_result.json"
    continue
  fi
  if [ -f "${out_dir}/remote_articulation_status.json" ] && \
     "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import json
from pathlib import Path
status = json.loads(Path("${out_dir}/remote_articulation_status.json").read_text())
raise SystemExit(0 if status.get("status") == "completed" else 1)
PY
  then
    echo "[$object_id] skip completed remote status ${out_dir}/remote_articulation_status.json"
    continue
  fi

  if [ ! -f "$episode" ]; then
    echo "[$object_id] missing episode: $episode" | tee -a "$log"
    failures+=("$object_id:missing_episode")
    continue
  fi

  frame_index=$(PYTHONPATH=src "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

data = json.loads(Path("$episode").read_text())
frames = data["frames"]
secondary_start_s = float(data.get("metadata", {}).get("staged_control", {}).get("secondary_start_s", 1.2))
deadline = secondary_start_s - 0.15
candidates = []
for i, frame in enumerate(frames):
    t = float(frame.get("time_s", i / 60.0))
    if t <= deadline and frame.get("joint_position_hint") is not None:
        candidates.append((i, abs(float(frame["joint_position_hint"])), t))
if not candidates:
    candidates = [
        (i, abs(float(frame.get("joint_position_hint") or 0.0)), float(frame.get("time_s", i / 60.0)))
        for i, frame in enumerate(frames)
    ]
print(max(candidates, key=lambda item: item[1])[0])
PY
)

  echo "[$object_id] frame=$frame_index view=1 -> $out_dir" | tee "$log"
  args=(
    -m rgbd_urdf_mvp remote-articulate-generate
    --server-url "$REMOTE_ARTICULATION_SERVER_URL"
    --episode "$episode"
    --generation-image-output-dir "$input_dir"
    --frame-index "$frame_index"
    --view-indices 1
    --image-views front
    --mask-source object
    --background transparent
    --output-dir "$out_dir"
    --api-token "$REMOTE_ARTICULATION_API_TOKEN"
    --face-count 20000
    --particulate-up-dir Y
    --particulate-target-faces 30000
    --particulate-global-points 10000
    --particulate-num-points 10000
    --particulate-no-strict
    --timeout-s 3600
  )
  if [ "$SKIP_DOWNLOAD" != "0" ]; then
    args+=(--remote-skip-download)
  fi

  PYTHONPATH=src "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    failures+=("$object_id:rc=$rc")
    echo "[$object_id] FAILED rc=$rc" | tee -a "$log"
  else
    echo "[$object_id] completed" | tee -a "$log"
  fi
done < "$BATCH_CONFIG"

if [ "${#failures[@]}" -gt 0 ]; then
  {
    echo "Failures:"
    printf '%s\n' "${failures[@]}"
  } | tee "$OUTPUT_ROOT/_logs/failures.txt" >&2
  exit 1
fi
