#!/usr/bin/env bash
# =============================================================================
# submit_era5_slurm.sh — SLURM array submit for PF Stage-2 ERA-5 co-location
# =============================================================================
#
# Stage 2 reads the feature table built by Stage 1 and co-locates ARCO ERA-5
# (centroid + box stats + 0-1/3/6 km shear) at each feature, writing a per-orbit
# table under {root}/era5. It is embarrassingly parallel + idempotent, so — like
# Stage 1 — it runs as a per-day SLURM ARRAY (one task/day), NOT via Dask:
#   * each day-task runs scripts/add_era5.py for that day, which groups that
#     day's features by ERA-5 hour and fetches each hour ONCE (GCS), matching all
#     features in it in memory;
#   * <=PF_MAX_NODES nodes run at once via the %N array throttle.
# Dask (feng_tracking's approach) buys dynamic load-balancing across uneven
# chunks; our per-day partition is uniform and GCS-bound, so a static array is
# simpler and more robust (no long-lived scheduler/driver).
#
# USAGE
#   scripts/submit_era5_slurm.sh MISSION START END
#
# Tunables (env):
#   PF_MAX_NODES   nodes at once (%N throttle)            default 4
#   PF_ERA5_WORKERS  hour-batch workers per node (GCS)    default 16
#   PF_MEM         per-node memory                        default 64G
#   PF_OUT_ROOT    dataset root (features in, era5 out)   default /data/scratch/a/snesbitt/pf_db_dec1997
#
# CHAIN AFTER STAGE 1:
#   JID=$(scripts/submit_pf_slurm.sh TRMM 1997-12-15 1997-12-31 | awk '/array job/{print $NF}')
#   sbatch --dependency=afterok:${JID} ...   # or just run this after Stage 1 completes
# =============================================================================
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 MISSION START END   (dates YYYY-MM-DD, inclusive)" >&2
    exit 2
fi
MISSION="$1"; START="$2"; END="$3"

MAX_NODES="${PF_MAX_NODES:-4}"
WORKERS="${PF_ERA5_WORKERS:-16}"
MEM="${PF_MEM:-64G}"
OUT_ROOT="${PF_OUT_ROOT:-/data/scratch/a/snesbitt/pf_db_dec1997}"

START_EPOCH=$(date -u -d "$START" +%s); END_EPOCH=$(date -u -d "$END" +%s)
if (( END_EPOCH < START_EPOCH )); then echo "error: END before START" >&2; exit 2; fi
N_DAYS=$(( (END_EPOCH - START_EPOCH) / 86400 + 1 )); LAST_IDX=$(( N_DAYS - 1 ))

RUNDIR="${PF_RUNDIR:-$PWD/slurm_runs}"; mkdir -p "$RUNDIR"
SBATCH_SCRIPT="$RUNDIR/era5_${MISSION}_${START}_${END}.sbatch"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=era5_${MISSION}
#SBATCH --array=0-${LAST_IDX}%${MAX_NODES}
#SBATCH --cpus-per-task=20
#SBATCH --mem=${MEM}
#SBATCH --time=04:00:00
#SBATCH --output=${RUNDIR}/era5_${MISSION}_%A_%a.out
#SBATCH --error=${RUNDIR}/era5_${MISSION}_%A_%a.err
#SBATCH --partition=node
#SBATCH --account=snesbitt-group
#SBATCH --constraint=g20

set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf

# This task's day, and the EXCLUSIVE end (add_era5 filters time in [start,end)).
DAY=\$(date -u -d "${START} + \${SLURM_ARRAY_TASK_ID} days" +%F)
NEXT=\$(date -u -d "\${DAY} + 1 day" +%F)
echo "[\$(date -u +%FT%TZ)] task \${SLURM_ARRAY_TASK_ID} -> ERA-5 ${MISSION} \${DAY}"

cd "${REPO_ROOT}"
python scripts/add_era5.py "${MISSION}" \\
    --start "\${DAY}" --end "\${NEXT}" \\
    --root "${OUT_ROOT}" \\
    --workers ${WORKERS}
EOF

echo "Generated: $SBATCH_SCRIPT"
DEP="${PF_DEPENDENCY:-}"   # e.g. afterok:<stage1_jobid> to chain after the radar stage
echo "Submitting: ${N_DAYS} day(s), array 0-${LAST_IDX}%${MAX_NODES}, ${WORKERS} workers/node, ${MEM}, root=${OUT_ROOT}${DEP:+, dependency=$DEP}"
OUT=$(sbatch ${DEP:+--dependency=$DEP} "$SBATCH_SCRIPT"); echo "$OUT"
echo "Submitted array job $(awk '{print $NF}' <<<"$OUT")"
