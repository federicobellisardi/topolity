#!/usr/bin/env bash
# Slurm entry point for python/barcelona_transform_multimodal.py.
# `run` parses every option after the job name as its own, so the job takes
# positional arguments only:
#   transform_multimodal_job.sh <city> <task index>    one embedding
#   transform_multimodal_job.sh <city> merge           merge + scenarios
# The OD source is taken from $OD_SOURCE (twitter by default); transform_multimodal_mitma_job.sh sets it
# to mitma, because `run` appends the array parameter after the positional arguments.
set -euo pipefail
# Repository root: TOPOLITY_ROOT, else the parent of this script, else the Slurm submission directory
# (sbatch runs a copy of job scripts from its spool directory; the submit scripts cd to the root first).
_topolity_root() {
  local d
  for d in "${TOPOLITY_ROOT:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
    [[ -n "$d" && -f "$d/python/study_config.py" ]] && { echo "$d"; return 0; }
  done
  echo "cannot locate the topolity repository: set TOPOLITY_ROOT" >&2; return 1
}
ROOT="$(_topolity_root)"
# Machine-specific settings (PY, RUNLOG), not versioned: see pipeline_production/local.env.example
if [[ -f "$ROOT/pipeline_production/local.env" ]]; then source "$ROOT/pipeline_production/local.env"; fi
PY="${PY:-python}"
CITY="$1"; WHAT="$2"; OD="${OD_SOURCE:-twitter}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$ROOT"
if [[ "$WHAT" == "merge" ]]; then
  exec "$PY" -u python/barcelona_transform_multimodal.py --city "$CITY" --od "$OD" --merge
fi
exec "$PY" -u python/barcelona_transform_multimodal.py --city "$CITY" --od "$OD" --task "$WHAT"
