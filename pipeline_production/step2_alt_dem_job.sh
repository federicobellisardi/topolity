#!/usr/bin/env bash
# Slurm entry point for python/step2_alt_dem.py (positional arguments only, see transform_multimodal_job.sh):
#   step2_alt_dem_job.sh <city> <task index>     one embedding (0 = observed)
#   step2_alt_dem_job.sh <city> merge            merged table + comparison with SRTM
# DEM_TAG (default 3dep13) selects data/data_processed/<city>/dem/<city>_dem_<tag>.tif.
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
CITY="$1"; WHAT="$2"; TAG="${DEM_TAG:-3dep13}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$ROOT"
if [[ "$WHAT" == "merge" ]]; then
  exec "$PY" -u python/step2_alt_dem.py --city "$CITY" --dem-tag "$TAG" --merge
fi
exec "$PY" -u python/step2_alt_dem.py --city "$CITY" --dem-tag "$TAG" --task "$WHAT"
