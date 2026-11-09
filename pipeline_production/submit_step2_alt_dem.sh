#!/usr/bin/env bash
set -euo pipefail

# DEM sensitivity test (USGS 3DEP bare earth instead of SRTM) for Chicago and Dallas: step 2 on the
# existing step-1 graphs, one Slurm task per embedding. Task 0 (observed embedding) runs first, the other
# embeddings run as an array after it, then one merge job writes the table and the comparison with SRTM:
#
#   DRY_RUN=1 CITY=chicago bash pipeline_production/submit_step2_alt_dem.sh     # print only
#   DRY_RUN=0 CITY=chicago bash pipeline_production/submit_step2_alt_dem.sh
#   DRY_RUN=0 CITY=dallas  bash pipeline_production/submit_step2_alt_dem.sh
#
# Needs data/data_processed/<city>/dem/<city>_dem_3dep13.tif (python/dem_3dep.py). Outputs go to
# output/step2_alt_dem/<city>/3dep13/; the SRTM results are not touched. Tasks resume, so a resubmission
# only fills the gaps.

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
SCRIPT="$ROOT/python/step2_alt_dem.py"
JOB="$ROOT/pipeline_production/step2_alt_dem_job.sh"
CITY="${CITY:?set CITY=chicago or CITY=dallas}"
TAG="${DEM_TAG:-3dep13}"
DRY_RUN="${DRY_RUN:-1}"
T_TASK="${T_TASK:-24:00}"; M_TASK="${M_TASK:-48}"; C_TASK="${C_TASK:-1}"
T_MERGE="${T_MERGE:-1:00}"; M_MERGE="${M_MERGE:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-40}"
export DEM_TAG="$TAG" OMP_NUM_THREADS="$C_TASK" OPENBLAS_NUM_THREADS="$C_TASK"
cd "$ROOT"

for f in "$PY" "$SCRIPT" "$JOB" "data/data_processed/$CITY/dem/${CITY}_dem_${TAG}.tif" \
         "data/data_processed/$CITY/graphs_fine_grid/fine_grid_gravitational_work.csv"; do
  [[ -e "$f" ]] || { echo "[abort] missing $f" >&2; exit 1; }
done
if [[ "$DRY_RUN" != "1" ]]; then
  command -v run >/dev/null || { echo "[abort] 'run' not in PATH on $(hostname)" >&2; exit 1; }
fi

N=$("$PY" "$SCRIPT" --city "$CITY" --list 2>/dev/null | grep -cE '^[0-9]+ graph_')
[[ "$N" -gt 1 ]] || { echo "[abort] no transformed embeddings listed for $CITY" >&2; exit 1; }
LOGDIR="$ROOT/logs/step2_alt_dem/$CITY"
mkdir -p "$LOGDIR"
PARAMS="$(seq -s ' ' 1 $((N - 1)))"
echo "[info] $CITY: observed + $((N - 1)) embeddings with DEM $TAG; logs in $LOGDIR"

T0=(run -t "$T_TASK" -m "$M_TASK" -c "$C_TASK" -j "dem0_$CITY" -o "$LOGDIR/task0_%j.out" -e "$LOGDIR/task0_%j.err"
    "$JOB" "$CITY" 0)
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $(printf '%q ' "${T0[@]}")"
  echo "[dry-run] then array: run -t $T_TASK -m $M_TASK -c $C_TASK -d <task0 id> --array $((N - 1)):$MAX_PARALLEL --params '1..$((N - 1))' -j demA_$CITY $JOB $CITY"
  echo "[dry-run] then: run -t $T_MERGE -m $M_MERGE -d <array id> -j demM_$CITY $JOB $CITY merge"
  exit 0
fi
out0="$("${T0[@]}" 2>&1)"; echo "$out0"
J0="$(grep -oE '[0-9]{5,}' <<<"$out0" | tail -1 || true)"
[[ -n "$J0" ]] || { echo "[abort] could not read the task-0 job id" >&2; exit 1; }
outA="$(run -t "$T_TASK" -m "$M_TASK" -c "$C_TASK" -d "$J0" --array "$((N - 1)):$MAX_PARALLEL" --params "$PARAMS" \
        -j "demA_$CITY" -o "$LOGDIR/task_%A_%a.out" -e "$LOGDIR/task_%A_%a.err" "$JOB" "$CITY" 2>&1)"; echo "$outA"
JA="$(grep -oE '[0-9]{5,}' <<<"$outA" | tail -1 || true)"
[[ -n "$JA" ]] || { echo "[abort] could not read the array job id; merge not submitted" >&2; exit 1; }
run -t "$T_MERGE" -m "$M_MERGE" -d "$JA" -j "demM_$CITY" -o "$LOGDIR/merge_%j.out" -e "$LOGDIR/merge_%j.err" \
    "$JOB" "$CITY" merge
echo "[ok] task 0 = $J0, array = $JA (after task 0), merge after the array"
