#!/usr/bin/env bash
# =============================================================================
# submit_grid_slurm.sh — SLURM submit for Part B gridded rain climatology
# =============================================================================
# Post-hoc product: grids the PF near-surface rain into 0.05 deg bins, stratified
# by feature size / 20-dBZ echo-top / rain type / month-of-year, divided by the
# per-orbit views denominator. One CF-NetCDF + one sparse-joint Parquet PER
# MISSION (scripts/grid_climatology.py). The 12 months-of-year are parallelized
# INTRA-NODE (a spawn Pool), so this is a SINGLE job per mission — not an array.
#
# USAGE
#   scripts/submit_grid_slurm.sh MISSION
#     MISSION = GPM | TRMM
#
# Tunables (env):
#   PF_MODE         crossproduct | marginal                default crossproduct
#   PF_WORKERS      month workers (Pool, <=12)             default 12
#   PF_DUCKDB_MEM   DuckDB memory_limit per worker         default 16GB
#   PF_DUCKDB_THREADS  DuckDB threads per worker           default 2
#   PF_MEM          per-node memory                        default 256G (crossproduct)
#   PF_CPUS         cpus-per-task                          default 24
#   PF_PARTITION / PF_CONSTRAINT                           default seseml / j48
#   PF_OUT_ROOT     dataset root (in)                      default /data/scratch/a/snesbitt/pf_db
#   PF_GRID_OUT     product out dir                        default {OUT_ROOT}/grid
#   PF_DEPENDENCY   afterok:<radar_jobid> to chain after the rebuild+views land
#   PF_DRYRUN=1     print the generated sbatch, do NOT submit
# =============================================================================
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 MISSION(GPM|TRMM)" >&2; exit 2; }
MISSION="$1"

WORKERS="${PF_WORKERS:-12}"
DUCKDB_MEM="${PF_DUCKDB_MEM:-16GB}"
DUCKDB_THREADS="${PF_DUCKDB_THREADS:-2}"
MEM="${PF_MEM:-192G}"
CPUS="${PF_CPUS:-48}"
PARTITION="${PF_PARTITION:-seseml}"
CONSTRAINT="${PF_CONSTRAINT:-j48}"
OUT_ROOT="${PF_OUT_ROOT:-/data/scratch/a/snesbitt/pf_db}"
GRID_OUT="${PF_GRID_OUT:-${OUT_ROOT}/grid}"
DEP="${PF_DEPENDENCY:-}"

RUNDIR="${PF_RUNDIR:-$PWD/slurm_runs}"; mkdir -p "$RUNDIR"
SBATCH="$RUNDIR/grid_${MISSION}.sbatch"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat > "$SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=grid_${MISSION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=08:00:00
#SBATCH --output=${RUNDIR}/grid_${MISSION}_%j.out
#SBATCH --error=${RUNDIR}/grid_${MISSION}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=snesbitt-group
#SBATCH --constraint=${CONSTRAINT}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"; conda activate pf
source /data/keeling/a/snesbitt/.spaceborne_minio.env   # MinIO creds for the icechunk write
echo "[\$(date -u +%FT%TZ)] grid ${MISSION} (icechunk) workers=${WORKERS}"
cd "${REPO}"
export PF_DUCKDB_MEM="${DUCKDB_MEM}" PF_DUCKDB_THREADS="${DUCKDB_THREADS}"
python scripts/grid_climatology.py "${MISSION}" \\
    --root "${OUT_ROOT}" \\
    --out  "${GRID_OUT}" \\
    --workers ${WORKERS}
EOF

echo "MISSION=$MISSION (icechunk) workers=$WORKERS duckdb_mem=$DUCKDB_MEM part=$PARTITION feat=$CONSTRAINT ${MEM} root=$OUT_ROOT out=$GRID_OUT${DEP:+ dep=$DEP}"
if [[ "${PF_DRYRUN:-0}" == 1 ]]; then
  echo "--- DRYRUN: generated sbatch ---"; cat "$SBATCH"; exit 0
fi
OUT=$(sbatch ${DEP:+--dependency=$DEP} "$SBATCH"); echo "$OUT"
echo "Submitted job $(awk '{print $NF}' <<<"$OUT")"
