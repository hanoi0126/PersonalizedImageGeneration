if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

git config --global --add safe.directory /workspace

CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 1 \
    fastcomposer/train_with_face_loss.py
