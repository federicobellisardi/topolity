#!/usr/bin/env bash
# =============================================================================
# run_weekend.sh
# Pipeline completa: step1 → step2 → supplementary per le città da ricostruire
# + Atlanta/Santiago con angoli random.
#
# Uso:
#   bash pipeline_production/run_weekend.sh           # lancia su SLURM via runlog
#   bash pipeline_production/run_weekend.sh --dry-run # stampa comandi senza lanciare
# =============================================================================
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Percorsi ──────────────────────────────────────────────────────────────────
ROOT="/home/fbellisardi/code/topolity"
DATA_ROOT="$ROOT/data/data_processed"
STEP1="$ROOT/pipeline_production/dem_extractor_fine_grid.py"
STEP2="$ROOT/pipeline_production/fine_grid_gravitational_work.py"
SUPP="$ROOT/supplementary/python/build_gravitational_supplementary.py"

# ── Risorse SLURM ─────────────────────────────────────────────────────────────
# -m 96: con land_mask locale la RAM è molto più gestibile (fix nel codice),
# ma il DEM tree per FUA grandi (Chicago, Toronto) può pesare 5-15 GB da solo.
S1_TIME="72:00" ; S1_MEM=96  ; S1_CPU=4   # step 1
S2_TIME="8:00"  ; S2_MEM=32  ; S2_CPU=4   # step 2
SB_TIME="2:00"  ; SB_MEM=32  ; SB_CPU=1   # supplementary builder

# ── Parametri step 1 ─────────────────────────────────────────────────────────
S1_COMMON=(
    --use-fua
    --step-meters 500
    --num-points 5
    --rotation-angles -20 -15 -10 -5 5 10 15 20
    --ns-scale-factors 1.02 1.05 1.08 1.12
    --ew-scale-factors 1.02 1.05 1.08 1.12
    --land-check full
    --seed 42
    --resume
)

# Angoli extra per santiago e atlanta (configurazioni "random" per il notebook)
EXTRA_ANGLES=(--extra-translation-angles 30 45 135 225 315)

# ── Città ─────────────────────────────────────────────────────────────────────
CITIES_STD=(amsterdam bruxelles milan bogota bandung buenosaires toronto chicago)
CITIES_EXTRA=(santiago atlanta)   # stesse opzioni + angoli random

# =============================================================================
# Helper: lancia un job e restituisce il job ID su stdout
# =============================================================================
submit() {
    local label="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] runlog $*" >&2
        echo "$((RANDOM + 10000))"
        return 0
    fi

    local raw
    raw=$(runlog "$@" 2>&1)
    printf '%s\n' "$raw" >&2

    # Estrae job ID: "Submitted batch job NNNN" oppure ultima sequenza 5+ cifre
    local jid
    jid=$(echo "$raw" | grep -oP '(?i)(?:submitted\s+batch\s+job\s+)\K\d+' | tail -1)
    if [[ -z "$jid" ]]; then
        jid=$(echo "$raw" | grep -oP '\b\d{5,}\b' | tail -1)
    fi
    if [[ -z "$jid" ]]; then
        echo "ERRORE [$label]: impossibile estrarre job ID da runlog" >&2
        echo "$raw" >&2
        exit 1
    fi
    echo "$jid"
}

# =============================================================================
# 0. Pulizia file stale (va fatta PRIMA di inviare qualsiasi job)
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [0/4] Pulizia file stale"
echo "════════════════════════════════════════════════"

# Santiago: FUA bounds == bbox bounds → stale detection non scatta automaticamente
echo "  → santiago: land_shp + graph_original + fine_grid..."
if [[ "$DRY_RUN" == "0" ]]; then
    SDIR="$DATA_ROOT/santiago"
    rm -f "$SDIR/land/santiago_clipped_land".{shp,dbf,cpg,prj,shx}
    rm -f "$SDIR/graphs/graph_original.pkl"
    find "$SDIR/graphs_fine_grid" -name 'graph_*.pkl'    -delete 2>/dev/null || true
    rm -f "$SDIR/graphs_fine_grid/fine_grid_stats.csv" \
          "$SDIR/graphs_fine_grid/fine_grid_gravitational_work.csv"
fi

# Toronto e Chicago: corrette con FUA ma lake mask non era attivo
# → eliminiamo i pkl così vengono ri-calcolati con il lake check locale
for city in toronto chicago; do
    echo "  → $city: fine_grid pkl (lake mask fix)..."
    if [[ "$DRY_RUN" == "0" ]]; then
        CDIR="$DATA_ROOT/$city"
        find "$CDIR/graphs_fine_grid" -name 'graph_*.pkl' -delete 2>/dev/null || true
        rm -f "$CDIR/graphs_fine_grid/fine_grid_stats.csv" \
              "$CDIR/graphs_fine_grid/fine_grid_gravitational_work.csv"
    fi
done

echo "  Pulizia completata."

# =============================================================================
# 1. Step 1 — tutte le città (10 job SLURM in parallelo)
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [1/4] Step 1 — ricostruzione grafi"
echo "════════════════════════════════════════════════"

declare -A S1_IDS

for city in "${CITIES_STD[@]}"; do
    jid=$(submit "fg_${city}" \
        -t "$S1_TIME" -m "$S1_MEM" -c "$S1_CPU" -j "fg_${city}" \
        python -u "$STEP1" --city "$city" "${S1_COMMON[@]}")
    S1_IDS[$city]=$jid
    echo "  fg_${city} → job $jid"
done

for city in "${CITIES_EXTRA[@]}"; do
    jid=$(submit "fg_${city}" \
        -t "$S1_TIME" -m "$S1_MEM" -c "$S1_CPU" -j "fg_${city}" \
        python -u "$STEP1" --city "$city" "${S1_COMMON[@]}" "${EXTRA_ANGLES[@]}")
    S1_IDS[$city]=$jid
    echo "  fg_${city} (+ angoli random) → job $jid"
done

# =============================================================================
# 2. Step 2 — parte solo quando il suo step1 è finito (-d JOB_ID)
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [2/4] Step 2 — calcolo lavoro gravitazionale"
echo "════════════════════════════════════════════════"

declare -A S2_IDS

ALL_CITIES=("${CITIES_STD[@]}" "${CITIES_EXTRA[@]}")
for city in "${ALL_CITIES[@]}"; do
    s1_jid="${S1_IDS[$city]}"
    jid=$(submit "fgw_${city}" \
        -t "$S2_TIME" -m "$S2_MEM" -c "$S2_CPU" -j "fgw_${city}" \
        -d "$s1_jid" \
        python "$STEP2" \
            --data-root "$DATA_ROOT" \
            --city "$city" \
            --resume)
    S2_IDS[$city]=$jid
    echo "  fgw_${city} → job $jid  (after $s1_jid)"
done

# =============================================================================
# 3. Supplementary builder — parte dopo TUTTI gli step2
#    SLURM supporta afterok:JOB1:JOB2:...; se runlog usa la stessa sintassi
#    il builder partirà automaticamente. In caso contrario: vedi nota finale.
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [3/4] Supplementary builder"
echo "════════════════════════════════════════════════"

all_s2_ids=$(printf '%s:' "${S2_IDS[@]}")
all_s2_ids="${all_s2_ids%:}"   # rimuove ':' finale

sb_jid=$(submit "supp_builder" \
    -t "$SB_TIME" -m "$SB_MEM" -c "$SB_CPU" -j "supp_builder" \
    -d "$all_s2_ids" \
    python "$SUPP" --only all)
echo "  supp_builder → job $sb_jid"
echo "  dipende da: $all_s2_ids"

# =============================================================================
# 4. Riepilogo
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [4/4] Riepilogo"
echo "════════════════════════════════════════════════"
echo ""
printf "  %-20s %s\n" "STEP 1" "JOB ID"
for city in "${ALL_CITIES[@]}"; do
    printf "  %-20s %s\n" "fg_${city}" "${S1_IDS[$city]}"
done
echo ""
printf "  %-20s %s\n" "STEP 2" "JOB ID"
for city in "${ALL_CITIES[@]}"; do
    printf "  %-20s %s\n" "fgw_${city}" "${S2_IDS[$city]}"
done
echo ""
echo "  SUPPLEMENTARY: $sb_jid"
echo ""
echo "  Monitor:"
echo "    watch -n 30 'squeue -u \$USER --format=\"%.8i %.15j %.8T %.10M %R\"'"
echo ""
echo "  Se supp_builder non parte (dipendenza multipla non supportata):"
echo "    runlog -t $SB_TIME -m $SB_MEM -c $SB_CPU -j supp_builder \\"
echo "      python $SUPP --only all"
