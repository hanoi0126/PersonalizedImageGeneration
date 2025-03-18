#!/bin/sh
#PJM -L elapse=0:30:00
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -g gb20
#PJM -j

module load gcc/8.3.1
module load python/3.10.13
module load cuda/11.8

source venv/bin/activate
source jobs/import-env.sh .env

# create log directory
LOG_DIR="logs/$(date '+%Y-%m-%d/%H-%M-%S')"
mkdir -p ${LOG_DIR}
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME

# record job information
cat <<EOF >> ${LOG_DIR}/job_output.log
=== Job Information ===
Job ID: ${PJM_JOBID}
Job started at $(date)

=== NVIDIA-SMI Output ===
EOF
nvidia-smi >> ${LOG_DIR}/job_output.log

# run python script
cat <<EOF >> ${LOG_DIR}/job_output.log

=== Main Output ===
EOF

export PYTHONPATH="$PWD:$PYTHONPATH"

python fastcomposer/batch_infer.py >> ${LOG_DIR}/job_output.log 2>&1

# record job information
cat <<EOF >> ${LOG_DIR}/job_output.log

=== Job Information ===
Job finished at $(date)
EOF