#!/usr/bin/env bash
# Slurm entry point of the multimodal transformation test with MITMA OD flows, trips >= 2 km (SI Note 9).
# Same positional arguments as transform_multimodal_job.sh.
set -euo pipefail
export OD_SOURCE=mitma_ge2km
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
exec "$ROOT/pipeline_production/transform_multimodal_job.sh" "$@"
