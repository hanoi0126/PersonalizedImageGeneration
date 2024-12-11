#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -g gb20
#PJM -j

module load gcc/8.3.1
module load python/3.10.13
module load cuda/11.8

source venv/bin/activate
source jobs/import-env.sh .env

# create log directory
LOG_DIR="outputs/$(date '+%Y-%m-%d/%H-%M-%S')"
mkdir -p ${LOG_DIR}

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

PYTHONPATH="$PWD:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 1 \
    fastcomposer/train_with_face_loss.py >> ${LOG_DIR}/job_output.log 2>&1

# record job information
cat <<EOF >> ${LOG_DIR}/job_output.log

=== Job Information ===
Job finished at $(date)
EOF