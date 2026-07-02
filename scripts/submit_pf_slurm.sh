#!/usr/bin/env bash
# =============================================================================
# submit_pf_slurm.sh — SLURM array submit wrapper for the PF radar+imager pipeline
# =============================================================================
#
# Splits a [START, END] date window into per-day chunks and submits a single
# sbatch ARRAY job — one array task per day — each running
# scripts/run_orbits_parallel.py for that day with 8 download-bound workers and
# a tmpfs download cache (/dev/shm). Concurrency across the array is capped at
# 8 simultaneous tasks (8 nodes) via the `%8` array throttle.
#
# USAGE
# -----
#   scripts/submit_pf_slurm.sh MISSION START END
#
#   MISSION  GPM | TRMM
#   START    YYYY-MM-DD (inclusive)
#   END      YYYY-MM-DD (inclusive)
#
# Example:
#   scripts/submit_pf_slurm.sh GPM 2018-07-01 2018-07-31
#
# The script PRINTS the generated sbatch script and the submit command, then
# submits it. Edit the #SBATCH directives below (partition / account / mem /
# time) for your cluster — they are a template.
#
# STAGE-2 (ERA-5) CHAINING
# ------------------------
# Capture the array job id from this script's stdout (it echoes
# "Submitted array job <JOBID>"), then chain a Stage-2 ERA-5 job that runs only
# after every array task succeeds:
#
#   JID=$(scripts/submit_pf_slurm.sh GPM 2018-07-01 2018-07-31 | awk '/array job/{print $NF}')
#   sbatch --dependency=afterok:${JID} scripts/era5_stage2.sh GPM 2018-07-01 2018-07-31
#
# `afterok:<JID>` (without an array index) waits for the WHOLE array to finish
# successfully, so Stage-2 sees a complete Stage-1 PF database.
# =============================================================================

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 MISSION START END   (dates YYYY-MM-DD, inclusive)" >&2
    exit 2
fi

MISSION="$1"
START="$2"
END="$3"

# --- Tunables (env-overridable) ---------------------------------------------
# PF_MAX_NODES : max array tasks (= nodes) running at once (the %N throttle).
# PF_WORKERS   : download-bound spawn-Pool workers per node.
# PF_MEM       : per-node memory request (covers /dev/shm tmpfs cache + working set;
#                ~8 workers peak ≈ 16-20 GB, so 64G is ample on the 128 GB g20 nodes).
MAX_NODES="${PF_MAX_NODES:-4}"
WORKERS="${PF_WORKERS:-8}"
MEM="${PF_MEM:-64G}"

# --- Build the per-day chunk list -------------------------------------------
# One array index per day. Each index N maps to START + N days. We compute the
# number of days inclusive; the array task derives its own date from the index.
START_EPOCH=$(date -u -d "$START" +%s)
END_EPOCH=$(date -u -d "$END" +%s)
if (( END_EPOCH < START_EPOCH )); then
    echo "error: END ($END) is before START ($START)" >&2
    exit 2
fi
N_DAYS=$(( (END_EPOCH - START_EPOCH) / 86400 + 1 ))
LAST_IDX=$(( N_DAYS - 1 ))

# Where logs / generated scripts go.
RUNDIR="${PF_RUNDIR:-$PWD/slurm_runs}"
mkdir -p "$RUNDIR"
SBATCH_SCRIPT="$RUNDIR/pf_${MISSION}_${START}_${END}.sbatch"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Generate the array sbatch script ---------------------------------------
cat > "$SBATCH_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=pf_${MISSION}
#SBATCH --array=0-${LAST_IDX}%${MAX_NODES}   # one task/day; <=${MAX_NODES} nodes at once
#SBATCH --cpus-per-task=20               # download-bound workers + headroom
#SBATCH --mem=${MEM}                     # covers /dev/shm tmpfs cache + working set
#SBATCH --time=04:00:00                  # per-day wallclock; adjust as needed
#SBATCH --output=${RUNDIR}/pf_${MISSION}_%A_%a.out
#SBATCH --error=${RUNDIR}/pf_${MISSION}_%A_%a.err
# ---- keeling.earth.illinois.edu ----
#SBATCH --partition=node
#SBATCH --account=snesbitt-group
#SBATCH --constraint=g20                  # 20-CPU nodes (g20,m128 / g20,m256)
# -------------------------------------

set -euo pipefail

# Activate the pf conda environment.
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate pf

# Derive this task's day from the array index.
DAY=\$(date -u -d "${START} + \${SLURM_ARRAY_TASK_ID} days" +%F)
echo "[\$(date -u +%FT%TZ)] task \${SLURM_ARRAY_TASK_ID} -> ${MISSION} \${DAY}"

cd "${REPO_ROOT}"

# Per-day, per-node parallel run. tmpfs cache is self-bounding: each worker
# downloads to /dev/shm/pf_*, processes, and deletes it (see run_orbits_parallel).
python scripts/run_orbits_parallel.py "${MISSION}" \\
    --start "\${DAY}" --end "\${DAY}" \\
    --workers ${WORKERS} \\
    --download-dir /dev/shm \\
    --root "${PF_OUT_ROOT:-/data/scratch/a/snesbitt/pf_db_dec1997}" \\
    --failed-file "${RUNDIR}/failed_${MISSION}_\${DAY}.txt"
EOF

echo "Generated sbatch script: $SBATCH_SCRIPT"
echo "----------------------------------------------------------------------"
cat "$SBATCH_SCRIPT"
echo "----------------------------------------------------------------------"
echo "Submitting: sbatch $SBATCH_SCRIPT  (${N_DAYS} day(s), array 0-${LAST_IDX}%${MAX_NODES}, ${WORKERS} workers/node, ${MEM})"

OUT=$(sbatch "$SBATCH_SCRIPT")
echo "$OUT"
# Normalise to a stable, greppable line for Stage-2 chaining.
JID=$(awk '{print $NF}' <<<"$OUT")
echo "Submitted array job ${JID}"
