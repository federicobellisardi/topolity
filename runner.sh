#!/usr/bin/env bash
# =============================================================================
# run_weekend.sh
# Pipeline completa: step1 → step2 → supplementary per le città da ricostruire
# + Atlanta con angoli random.
#
# Uso:
#   bash runner.sh           # lancia tutto su SLURM via runlog
#   bash runner.sh --dry-run # stampa i comandi senza lanciare nulla
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
S1_TIME="72:00" ; S1_MEM=96  ; S1_CPU=4   # step 1: rebuild + varianti
S2_TIME="8:00"  ; S2_MEM=32  ; S2_CPU=4   # step 2: calcolo energia
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
# Città standard (nessun angolo extra)
CITIES_STD=(amsterdam bruxelles milan bogota bandung buenosaires toronto chicago)
# Città con angoli extra
CITIES_EXTRA=(santiago atlanta)

# =============================================================================
# Helper: lancia un job con runlog e restituisce il job ID su stdout
# =============================================================================
submit() {
    local label="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] runlog $*" >&2
        # Restituisce un ID fittizio incrementale in dry-run
        echo "$((RANDOM + 10000))"
        return 0
    fi

    local raw
    raw=$(runlog "$@" 2>&1)
    printf '%s\n' "$raw" >&2

    # Estrae il job ID SLURM: "Submitted batch job NNNN" oppure ultima sequenza di 5+ cifre
    local jid
    jid=$(echo "$raw" | grep -oP '(?i)(?:submitted\s+batch\s+job\s+)\K\d+' | tail -1)
    if [[ -z "$jid" ]]; then
        jid=$(echo "$raw" | grep -oP '\b\d{5,}\b' | tail -1)
    fi
    if [[ -z "$jid" ]]; then
        echo "ERRORE: impossibile estrarre job ID da runlog per '$label'" >&2
        echo "$raw" >&2
        exit 1
    fi
    echo "$jid"
}

# =============================================================================
# 0. Pulizia file stale PRIMA di inviare qualsiasi job
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [0/4] Pulizia file stale"
echo "════════════════════════════════════════════════"

# Santiago: stale detection non cattura bounds identici tra FUA e bbox
# → eliminiamo manualmente così il prossimo step1 parte da zero con FUA
echo "  → santiago: eliminazione land_shp + graph_original + fine_grid..."
if [[ "$DRY_RUN" == "0" ]]; then
    SDIR="$DATA_ROOT/santiago"
    rm -f "$SDIR/land/santiago_clipped_land".{shp,dbf,cpg,prj,shx}
    rm -f "$SDIR/graphs/graph_original.pkl"
    find "$SDIR/graphs_fine_grid" -name 'graph_*.pkl'    -delete 2>/dev/null || true
    rm -f "$SDIR/graphs_fine_grid/fine_grid_stats.csv" \
          "$SDIR/graphs_fine_grid/fine_grid_gravitational_work.csv"
    echo "     fatto."
fi

# Toronto e Chicago: corretti con FUA, ma lake mask non era attivo
# → eliminiamo i pkl così vengono ri-calcolati con il lake check aggiornato
for city in toronto chicago; do
    echo "  → $city: eliminazione fine_grid pkl (lake mask fix)..."
    if [[ "$DRY_RUN" == "0" ]]; then
        CDIR="$DATA_ROOT/$city"
        find "$CDIR/graphs_fine_grid" -name 'graph_*.pkl' -delete 2>/dev/null || true
        rm -f "$CDIR/graphs_fine_grid/fine_grid_stats.csv" \
              "$CDIR/graphs_fine_grid/fine_grid_gravitational_work.csv"
        echo "     fatto."
    fi
done

echo "  Pulizia completata."

# =============================================================================
# 1. Step 1 — tutte le città (parallele, ognuna su 4 core)
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [1/4] Step 1 — ricostruzione grafi"
echo "════════════════════════════════════════════════"

declare -A S1_IDS

# Città standard
for city in "${CITIES_STD[@]}"; do
    jid=$(submit "fg_${city}" \
        -t "$S1_TIME" -m "$S1_MEM" -c "$S1_CPU" -j "fg_${city}" \
        python -u "$STEP1" --city "$city" "${S1_COMMON[@]}")
    S1_IDS[$city]=$jid
    echo "  fg_${city} → job $jid"
done

# Santiago e Atlanta: step 1 + angoli random per il notebook
for city in "${CITIES_EXTRA[@]}"; do
    jid=$(submit "fg_${city}" \
        -t "$S1_TIME" -m "$S1_MEM" -c "$S1_CPU" -j "fg_${city}" \
        python -u "$STEP1" --city "$city" "${S1_COMMON[@]}" "${EXTRA_ANGLES[@]}")
    S1_IDS[$city]=$jid
    echo "  fg_${city} (+ extra angles) → job $jid"
done

# =============================================================================
# 2. Step 2 — per ogni città, parte solo quando il suo step1 è finito
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
    echo "  fgw_${city} → job $jid  (parte dopo job $s1_jid)"
done

# =============================================================================
# 3. Supplementary builder — parte quando TUTTI gli step2 sono finiti
#    runlog -d accetta JOB1:JOB2:... (sintassi SLURM afterok multiplo)
# =============================================================================
echo ""
echo "════════════════════════════════════════════════"
echo " [3/4] Supplementary builder"
echo "════════════════════════════════════════════════"

# Costruisce la lista "JOB1:JOB2:..." per la dipendenza multipla
all_s2_ids=$(printf '%s:' "${S2_IDS[@]}")
all_s2_ids="${all_s2_ids%:}"   # rimuove il ':' finale

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
echo " [4/4] Riepilogo job inviati"
echo "════════════════════════════════════════════════"
echo ""
echo "  STEP 1:"
for city in "${ALL_CITIES[@]}"; do
    echo "    fg_${city}:  ${S1_IDS[$city]}"
done
echo ""
echo "  STEP 2:"
for city in "${ALL_CITIES[@]}"; do
    echo "    fgw_${city}: ${S2_IDS[$city]}"
done
echo ""
echo "  SUPPLEMENTARY: $sb_jid"
echo ""
echo "  Monitoraggio:"
echo "    watch -n 30 'squeue -u \$USER --format=\"%.10i %.12j %.8T %.10M %.6D %R\"'"
echo ""
# Nota: se il supplementary builder fallisce per dipendenza multipla,
# lanciarlo manualmente con:
#   runlog -t 2:00 -m 32 -c 1 -j supp_builder python $SUPP --only all
echo "  Se supp_builder non parte (dipendenza multipla non supportata):"
echo "    runlog -t $SB_TIME -m $SB_MEM -c $SB_CPU -j supp_builder python $SUPP --only all"
