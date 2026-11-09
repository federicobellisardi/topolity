#!/usr/bin/env bash
set -euo pipefail

# Multimodal transformation test (SI Note 9): one Slurm array task per embedding,
# then one merge job that evaluates the scenarios. Run on a node where `run`
# and `sbatch` work:
#
#   DRY_RUN=1 CITY=madrid    bash pipeline_production/submit_transform_multimodal.sh   # print only
#   DRY_RUN=0 CITY=madrid    bash pipeline_production/submit_transform_multimodal.sh
#   DRY_RUN=0 CITY=barcelone bash pipeline_production/submit_transform_multimodal.sh
#   DRY_RUN=0 CITY=madrid OD=mitma_ge2km bash pipeline_production/submit_transform_multimodal.sh  # MITMA
#
# OD=mitma | mitma_ge2km need data/data_processed/<city>/<city>_mitma_model/1000_cells/
# od_matrix_mitma_weekday{,_ge2km}.csv (python/mitma_od.py); outputs go to output/transform_multimodal_<OD>/<city>.
# Tasks skip embeddings whose output already exists in
# output/transform_multimodal/<city>/sums, so a resubmission only fills the gaps.
# The OSM caches must exist before submitting (otherwise every task would
# download them at the same time).

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
SCRIPT="$ROOT/python/barcelona_transform_multimodal.py"
JOB="$ROOT/pipeline_production/transform_multimodal_job.sh"   # positional args only (see there)
CITY="${CITY:?set CITY=madrid or CITY=barcelone}"
DRY_RUN="${DRY_RUN:-1}"
OD="${OD:-twitter}"
case "$OD" in
  twitter) ;;
  mitma) JOB="$ROOT/pipeline_production/transform_multimodal_mitma_job.sh" ;;
  mitma_ge2km) JOB="$ROOT/pipeline_production/transform_multimodal_mitma_ge2km_job.sh" ;;
  *) echo "[abort] OD must be twitter, mitma or mitma_ge2km" >&2; exit 1 ;;
esac
T_TASK="${T_TASK:-4:00}";  M_TASK="${M_TASK:-16}"; C_TASK="${C_TASK:-2}"
T_MERGE="${T_MERGE:-4:00}"; M_MERGE="${M_MERGE:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-50}"

export OMP_NUM_THREADS="$C_TASK" OPENBLAS_NUM_THREADS="$C_TASK" MKL_NUM_THREADS="$C_TASK" NUMEXPR_NUM_THREADS="$C_TASK"
cd "$ROOT"

# ---- preflight -------------------------------------------------------------
for f in "$PY" "$SCRIPT" "$JOB" \
         "data/data_processed/$CITY/transit/osm_rail_routes_v2.json" \
         "data/data_processed/$CITY/transit/osm_bus_routes.json" \
         "data/data_processed/$CITY/graphs_fine_grid/fine_grid_gravitational_work.csv" \
         $([[ "$OD" != "twitter" ]] && echo "data/data_processed/$CITY/${CITY}_mitma_model/1000_cells/od_matrix_mitma_weekday${OD#mitma}.csv"); do
  [[ -e "$f" ]] || { echo "[abort] missing $f" >&2; exit 1; }
done
if [[ "$DRY_RUN" != "1" ]]; then
  command -v run >/dev/null || { echo "[abort] 'run' not in PATH on $(hostname)" >&2; exit 1; }
  command -v sbatch >/dev/null || { echo "[abort] 'sbatch' not in PATH on $(hostname)" >&2; exit 1; }
fi

N=$("$PY" "$SCRIPT" --city "$CITY" --od "$OD" --list 2>/dev/null | grep -cE '^[0-9]+ graph_')
[[ "$N" -gt 0 ]] || { echo "[abort] no embeddings listed for $CITY" >&2; exit 1; }
PARAMS="$(seq -s ' ' 0 $((N - 1)))"
LOGDIR="$ROOT/logs/transform_multimodal$([[ "$OD" != "twitter" ]] && echo "_$OD")/$CITY"
mkdir -p "$LOGDIR"
echo "[info] $CITY: $N embeddings, up to $MAX_PARALLEL tasks at a time; logs in $LOGDIR"

TASK_CMD=(run -t "$T_TASK" -m "$M_TASK" -c "$C_TASK" --array "$N:$MAX_PARALLEL" --params "$PARAMS"
          -j "mmT_${OD}_$CITY" -o "$LOGDIR/task_%A_%a.out" -e "$LOGDIR/task_%A_%a.err"
          "$JOB" "$CITY")
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $(printf '%q ' "${TASK_CMD[@]}")"
  echo "[dry-run] then: run -t $T_MERGE -m $M_MERGE -d <array job id> -j mmM_${OD}_$CITY $JOB $CITY merge"
  exit 0
fi

rc=0
out="$("${TASK_CMD[@]}" 2>&1)" || rc=$?
echo "$out"
[[ "$rc" -eq 0 ]] || { echo "[abort] run exited with code $rc; merge not submitted" >&2; exit 1; }
JOB_ID="$(grep -oE '[0-9]{5,}' <<<"$out" | tail -1 || true)"
[[ -n "$JOB_ID" ]] || { echo "[abort] could not read the array job id; merge not submitted" >&2; exit 1; }
run -t "$T_MERGE" -m "$M_MERGE" -d "$JOB_ID" -j "mmM_${OD}_$CITY" \
    -o "$LOGDIR/merge_%j.out" -e "$LOGDIR/merge_%j.err" \
    "$JOB" "$CITY" merge
echo "[ok] array job $JOB_ID submitted; merge waits for it (afterok)"
