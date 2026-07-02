#!/usr/bin/env bash
# =============================================================================
# submit_mission_slurm.sh — full-mission SLURM launcher (PER-MONTH array tasks)
# =============================================================================
# SLURM MaxArraySize=1001 forbids per-day arrays over a multi-year mission, so
# this submits ONE array task PER MONTH (TRMM ~209, GPM ~145 — both fit). Each
# task processes a whole month: radar -> run_orbits_parallel.py; era5 -> add_era5.py.
# Both are idempotent/resumable and use the orbit-start-day attribution, so month
# boundaries don't double-process.
#
# USAGE
#   scripts/submit_mission_slurm.sh STAGE MISSION START_YM END_YM
#     STAGE   = radar | era5 | grid
#     MISSION = GPM | TRMM
#     START_YM,END_YM = YYYY-MM (inclusive months)
#
# Tunables (env): PF_MAX_NODES(8) PF_WORKERS(16) PF_MEM(96G) PF_OUT_ROOT(/data/scratch/a/snesbitt/pf_db)
#                 PF_DEPENDENCY (e.g. afterok:<radar_jobid> to chain era5 after radar)
#                 PF_DRYRUN=1 prints the generated sbatch + month map, does NOT submit.
# =============================================================================
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 STAGE(radar|era5) MISSION START_YM END_YM" >&2; exit 2; }
STAGE="$1"; MISSION="$2"; START_YM="$3"; END_YM="$4"
[[ "$STAGE" == radar || "$STAGE" == era5 || "$STAGE" == grid ]] || { echo "STAGE must be radar|era5|grid" >&2; exit 2; }

MAX_NODES="${PF_MAX_NODES:-8}"; WORKERS="${PF_WORKERS:-16}"; MEM="${PF_MEM:-96G}"
CPUS="${PF_CPUS:-20}"; PARTITION="${PF_PARTITION:-node}"; CONSTRAINT="${PF_CONSTRAINT:-g20}"
OUT_ROOT="${PF_OUT_ROOT:-/data/scratch/a/snesbitt/pf_db}"
DEP="${PF_DEPENDENCY:-}"

# number of months inclusive
sy=${START_YM%-*}; sm=${START_YM#*-}; ey=${END_YM%-*}; em=${END_YM#*-}
N_MONTHS=$(( (10#$ey - 10#$sy)*12 + (10#$em - 10#$sm) + 1 ))
(( N_MONTHS > 0 )) || { echo "END before START" >&2; exit 2; }
LAST_IDX=$(( N_MONTHS - 1 ))
(( N_MONTHS <= 1001 )) || { echo "N_MONTHS=$N_MONTHS exceeds MaxArraySize 1001" >&2; exit 2; }

RUNDIR="${PF_RUNDIR:-$PWD/slurm_runs}"; mkdir -p "$RUNDIR"
SBATCH="$RUNDIR/${STAGE}_${MISSION}_${START_YM}_${END_YM}.sbatch"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRST="${START_YM}-01"

if [[ "$STAGE" == radar ]]; then
  CMD='python scripts/run_orbits_parallel.py "'"$MISSION"'" --start "${MS}" --end "${ME}" --workers '"$WORKERS"' --download-dir /dev/shm --root "'"$OUT_ROOT"'" --failed-file "'"$RUNDIR"'/failed_'"$MISSION"'_${MS}.txt"'
elif [[ "$STAGE" == grid ]]; then
  CMD='python scripts/grid_month.py "'"$MISSION"'" --start "${MS}" --end "${ME}" --workers '"$WORKERS"' --download-dir /dev/shm --root "'"$OUT_ROOT"'" --skip-existing --failed-file "'"$RUNDIR"'/failed_grid_'"$MISSION"'_${MS}.txt"'
else
  CMD='python scripts/add_era5.py "'"$MISSION"'" --start "${MS}" --end "${NEXT}" --root "'"$OUT_ROOT"'" --workers '"$WORKERS"
fi

cat > "$SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${STAGE}_${MISSION}
#SBATCH --array=0-${LAST_IDX}%${MAX_NODES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=08:00:00
#SBATCH --output=${RUNDIR}/${STAGE}_${MISSION}_%A_%a.out
#SBATCH --error=${RUNDIR}/${STAGE}_${MISSION}_%A_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --account=snesbitt-group
#SBATCH --constraint=${CONSTRAINT}
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"; conda activate pf
MS=\$(date -u -d "${FIRST} + \${SLURM_ARRAY_TASK_ID} months" +%Y-%m-%d)            # month start
ME=\$(date -u -d "\${MS} + 1 month - 1 day" +%Y-%m-%d)                              # month end (inclusive, radar)
NEXT=\$(date -u -d "\${MS} + 1 month" +%Y-%m-%d)                                    # next month start (era5 exclusive end)
echo "[\$(date -u +%FT%TZ)] ${STAGE} ${MISSION} task \${SLURM_ARRAY_TASK_ID}: \${MS} .. \${ME}"
cd "${REPO}"
${CMD}
EOF

echo "STAGE=$STAGE MISSION=$MISSION  ${N_MONTHS} months (array 0-${LAST_IDX}%${MAX_NODES}), ${WORKERS} workers/node, ${CPUS} cpus, part=$PARTITION, feat=$CONSTRAINT, ${MEM}, root=$OUT_ROOT${DEP:+, dep=$DEP}"
if [[ "${PF_DRYRUN:-0}" == 1 ]]; then
  echo "--- DRYRUN: generated sbatch ---"; cat "$SBATCH"
  echo "--- month map (first/last 3) ---"
  for i in 0 1 2 $((LAST_IDX-1)) $LAST_IDX; do echo "  idx $i -> $(date -u -d "${FIRST} + $i months" +%Y-%m)"; done
  exit 0
fi
OUT=$(sbatch ${DEP:+--dependency=$DEP} "$SBATCH"); echo "$OUT"
echo "Submitted array job $(awk '{print $NF}' <<<"$OUT")"
